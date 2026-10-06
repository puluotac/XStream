#include "hardware/kohzu/KohzuController.h"

#include <cmath>

#include <QDateTime>
#include <QRegularExpression>
#include <QStringList>
#include <QThread>

namespace ct
{

namespace
{

constexpr int kIdlePollMs = 1500;
// Position/status polling is only used for UI feedback and motion-release
// detection.  Keep this below the short step-move duration so a completed
// move is reported promptly, while still limiting the controller to a modest
// request rate during a long fly scan.
constexpr int kActivePollMs = 100;
constexpr qint64 kRdpTraceIntervalMs = 5000;
constexpr int kStablePollsBeforeInactive = 2;
constexpr qint64 kMinPositionStableMs = 160;
constexpr int kConnectTimeoutMs = 1500;
constexpr int kHandshakeTimeoutMs = 2500;
constexpr int kInitialAxisTimeoutMs = 2000;
constexpr int kPollAxisTimeoutMs = 700;
constexpr int kRdpRetryCount = 2;
constexpr qint64 kDefaultActiveReleaseMs = 15000;
constexpr qint64 kActiveReleaseMarginMs = 800;

enum class AxisMotionState
{
    Unknown,
    Moving,
    Idle
};

AxisMotionState parseAxisMotionState(const QString& statusRaw)
{
    // ARIES/LYNX responses are emitted in both forms seen in the field:
    // "C STR17 0 0 ..." and "C STR 17 0 0 ...".  The first numeric field
    // after the axis is the driving state: 0 stopped, 1 operating,
    // 2 feedback operating.
    static const QRegularExpression statusPattern(
        QStringLiteral("\\bC\\s+STR\\s*(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\b"),
        QRegularExpression::CaseInsensitiveOption);
    const QRegularExpressionMatch statusMatch = statusPattern.match(statusRaw);
    if (statusMatch.hasMatch())
    {
        const int drivingState = statusMatch.captured(2).toInt();
        if (drivingState == 0)
        {
            return AxisMotionState::Idle;
        }
        if (drivingState == 1 || drivingState == 2)
        {
            return AxisMotionState::Moving;
        }
        return AxisMotionState::Unknown;
    }

    const QString upper = statusRaw.toUpper();
    if (upper.contains(QStringLiteral("OPERATING"))
        || upper.contains(QStringLiteral("FB OPERATING")))
    {
        return AxisMotionState::Moving;
    }
    if (upper.contains(QStringLiteral("IDLE"))
        || upper.contains(QStringLiteral("READY"))
        || upper.contains(QStringLiteral("STOP"))
        || upper.contains(QStringLiteral("END")))
    {
        return AxisMotionState::Idle;
    }

    return AxisMotionState::Unknown;
}

QString responseTokenForCommand(const QString& command)
{
    const QString trimmed = command.trimmed();
    const int slash = trimmed.indexOf(QLatin1Char('/'));
    return slash >= 0 ? trimmed.left(slash) : trimmed;
}

QString findResponseForToken(const QString& responseText, const QString& token)
{
    if (token.isEmpty())
    {
        return responseText.trimmed();
    }

    const QRegularExpression responsePattern(
        QStringLiteral("(?:^|[\\r\\n])\\s*[CE]\\s+%1\\b[^\\r\\n]*")
            .arg(QRegularExpression::escape(token)),
        QRegularExpression::CaseInsensitiveOption);
    const QRegularExpressionMatch match = responsePattern.match(responseText);
    if (!match.hasMatch())
    {
        return {};
    }

    return match.captured(0).trimmed();
}

bool isKohzuBusyError(const QString& responseText)
{
    // ARIES error 803: next command was received before the previous response was completed.
    static const QRegularExpression busyPattern(
        QStringLiteral("\\bE\\s+\\S+\\s+803\\b"),
        QRegularExpression::CaseInsensitiveOption);
    return busyPattern.match(responseText).hasMatch();
}

bool isTransientSpeedTableError(const QString& responseText)
{
    // ARIES reports 605 when WTB arrives while the previous drive state is
    // still being released; 803 is its generic previous-command-busy error.
    static const QRegularExpression transientPattern(
        QStringLiteral("\\bE\\s+WTB\\d+\\s+(?:605|803)\\b"),
        QRegularExpression::CaseInsensitiveOption);
    return transientPattern.match(responseText).hasMatch() || isKohzuBusyError(responseText);
}

QList<int> toAxisList(const QVector<int>& axes)
{
    QList<int> axisList;
    axisList.reserve(axes.size());
    for (int axis : axes)
    {
        axisList.push_back(axis);
    }
    return axisList;
}

qint64 estimateMoveReleaseMs(qint64 distancePulse, int topSpeedPps, int accelTime10ms, int decelTime10ms)
{
    const double cruiseMs = topSpeedPps > 0
        ? (static_cast<double>(std::llabs(distancePulse)) / static_cast<double>(topSpeedPps)) * 1000.0
        : static_cast<double>(kDefaultActiveReleaseMs);
    const qint64 rampMs = static_cast<qint64>(std::max(0, accelTime10ms) + std::max(0, decelTime10ms)) * 10;
    return std::max<qint64>(700, static_cast<qint64>(std::ceil(cruiseMs)) + rampMs + kActiveReleaseMarginMs);
}

} // namespace

KohzuController::KohzuController(QObject* parent)
    : QObject(parent)
    , idlePollTimer_(this)
    , activePollTimer_(this)
{
    idlePollTimer_.setInterval(kIdlePollMs);
    activePollTimer_.setInterval(kActivePollMs);
    connect(&idlePollTimer_, &QTimer::timeout, this, &KohzuController::pollAllAxes);
    connect(&activePollTimer_, &QTimer::timeout, this, &KohzuController::pollActiveAxes);
}

void KohzuController::addAxis(int axis)
{
    if (!axes_.contains(axis))
    {
        axes_.push_back(axis);
    }
}

void KohzuController::clearAxes()
{
    const QList<int> activeAxes = activeAxes_.values();
    axes_.clear();
    activeAxes_.clear();
    lastActivePosition_.clear();
    lastKnownPosition_.clear();
    activeTargetPosition_.clear();
    activeStartedMs_.clear();
    activeReleaseAfterMs_.clear();
    activeStableCount_.clear();
    activeReadMissCount_.clear();
    activeLastMotionState_.clear();
    activeLastMotionState_.clear();
    speedTableCache_.clear();
    for (int axis : activeAxes)
    {
        emit axisActiveChanged(axis, false);
    }
}

void KohzuController::connectToController(const QString& host, quint16 port)
{
    closeConnection();

    socket_ = new QTcpSocket(this);
    socket_->connectToHost(host, port);
    if (!socket_->waitForConnected(kConnectTimeoutMs))
    {
        const QString error = socket_->errorString();
        delete socket_;
        socket_ = nullptr;
        emit connectionChanged(false);
        emit logMessage(QStringLiteral("Kohzu connect failed: %1").arg(error));
        return;
    }

    emit logMessage(QStringLiteral("Kohzu TCP connected: %1:%2, checking controller response...").arg(host).arg(port));

    QString idnError;
    QString idn;
    for (int attempt = 1; attempt <= 3 && idn.isEmpty(); ++attempt)
    {
        if (attempt > 1)
        {
            emit logMessage(QStringLiteral("Kohzu IDN retry %1/3...").arg(attempt));
            QThread::msleep(300);
        }
        idn = query(QStringLiteral("IDN"), &idnError, kHandshakeTimeoutMs);
    }

    bool controllerReady = !idn.isEmpty();
    if (controllerReady)
    {
        emit logMessage(QStringLiteral("Kohzu IDN: %1").arg(idn));
    }
    else
    {
        emit logMessage(QStringLiteral("Kohzu IDN failed after retries: %1").arg(idnError));
        const QHash<int, qint64> positions = readAxisPositions(toAxisList(axes_), kInitialAxisTimeoutMs);
        if (!positions.isEmpty())
        {
            emit logMessage(QStringLiteral("Kohzu controller verified by RDP: %1/%2 axes responded.")
                .arg(positions.size())
                .arg(axes_.size()));
            for (auto it = positions.constBegin(); it != positions.constEnd(); ++it)
            {
                emit axisUpdated(it.key(), it.value(), QStringLiteral("RDP"));
            }
            controllerReady = true;
        }
    }

    if (!controllerReady)
    {
        const QString finalError = idnError.isEmpty()
            ? QStringLiteral("controller did not respond")
            : idnError;
        emit logMessage(QStringLiteral("Kohzu protocol handshake failed; socket will be closed: %1").arg(finalError));
        closeConnection();
        return;
    }

    emit connectionChanged(true);
    emit logMessage(QStringLiteral("Kohzu connected: %1:%2").arg(host).arg(port));
    speedTableCache_.clear();
    pollAllAxesWithTimeout(kInitialAxisTimeoutMs);
    if (idlePollingEnabled_)
    {
        idlePollTimer_.start();
    }
}

void KohzuController::closeConnection()
{
    idlePollTimer_.stop();
    activePollTimer_.stop();
    const QList<int> activeAxes = activeAxes_.values();
    activeAxes_.clear();
    lastActivePosition_.clear();
    activeTargetPosition_.clear();
    activeStartedMs_.clear();
    activeReleaseAfterMs_.clear();
    activeStableCount_.clear();
    activeReadMissCount_.clear();
    activeLastMotionState_.clear();
    speedTableCache_.clear();
    for (int axis : activeAxes)
    {
        emit axisActiveChanged(axis, false);
    }

    if (socket_)
    {
        if (socket_->state() != QAbstractSocket::UnconnectedState)
        {
            socket_->disconnectFromHost();
            if (socket_->state() != QAbstractSocket::UnconnectedState)
            {
                socket_->waitForDisconnected(500);
            }
        }
        delete socket_;
        socket_ = nullptr;
    }
    emit connectionChanged(false);
}

void KohzuController::shutdown()
{
    closeConnection();
}

void KohzuController::jogRelativePulse(int axis, qint64 pulse, int speedTable, const QString& label)
{
    if (!isConnected())
    {
        emit commandFinished(label, false, QStringLiteral("Kohzu is not connected."));
        emit axisActiveChanged(axis, false);
        return;
    }

    const QString command = QStringLiteral("RPS%1/%2/%3/1").arg(axis).arg(speedTable).arg(pulse);
    QString error;
    const QString response = query(command, &error);
    const bool ok = !response.isEmpty() && response.startsWith(QLatin1Char('C'));
    emit commandFinished(label, ok, ok ? response : QStringLiteral("%1 %2").arg(error, response).trimmed());
    if (ok)
    {
        const bool hasKnownPosition = lastKnownPosition_.contains(axis);
        const qint64 targetPulse = hasKnownPosition ? lastKnownPosition_.value(axis) + pulse : 0;
        if (hasKnownPosition)
        {
            lastKnownPosition_[axis] = targetPulse;
            emit axisUpdated(axis, targetPulse, QStringLiteral("Estimated after RPS; waiting for RDP refresh."));
        }
        markAxisActive(axis, targetPulse, hasKnownPosition);
    }
    else
    {
        emit axisActiveChanged(axis, false);
    }
}

void KohzuController::moveAbsolutePulse(int axis, qint64 pulse, int speedTable, const QString& label)
{
    if (!isConnected())
    {
        emit commandFinished(label, false, QStringLiteral("Kohzu is not connected."));
        emit axisActiveChanged(axis, false);
        return;
    }

    QString idleStatus;
    if (!waitForAxisIdle(axis, 5000, &idleStatus))
    {
        emit commandFinished(label, false, QStringLiteral("axis %1 not idle before APS: %2").arg(axis).arg(idleStatus));
        emit axisActiveChanged(axis, false);
        return;
    }
    const QString command = QStringLiteral("APS%1/%2/%3/1").arg(axis).arg(speedTable).arg(pulse);
    QString error;
    const QString response = query(command, &error);
    const bool ok = !response.isEmpty() && response.startsWith(QLatin1Char('C'));
    emit commandFinished(label, ok, ok ? response : QStringLiteral("%1 %2").arg(error, response).trimmed());
    if (ok)
    {
        lastKnownPosition_[axis] = pulse;
        emit axisUpdated(axis, pulse, QStringLiteral("Estimated after APS; waiting for RDP refresh."));
        markAxisActive(axis, pulse, true);
    }
    else
    {
        emit axisActiveChanged(axis, false);
    }
}

void KohzuController::configureSpeedTable(int axis, int speedTable, int startSpeedPps, int topSpeedPps,
    int accelTime10ms, int decelTime10ms, int accelPattern, const QString& label)
{
    if (!isConnected())
    {
        emit commandFinished(label, false, QStringLiteral("Kohzu is not connected."));
        return;
    }

    if (speedTableCacheMatches(axis, speedTable, startSpeedPps, topSpeedPps,
        accelTime10ms, decelTime10ms, accelPattern))
    {
        emit commandFinished(label, true, QStringLiteral("speed table unchanged"));
        return;
    }

    QString idleStatus;
    if (!waitForAxisIdle(axis, 5000, &idleStatus))
    {
        emit commandFinished(label, false, QStringLiteral("axis %1 not idle before WTB: %2").arg(axis).arg(idleStatus));
        return;
    }

    QString error;
    const QString response = writeSpeedTable(axis, speedTable, startSpeedPps, topSpeedPps,
        accelTime10ms, decelTime10ms, accelPattern, &error);
    const bool ok = !response.isEmpty() && response.startsWith(QLatin1Char('C'));
    emit commandFinished(label, ok, ok ? response : QStringLiteral("%1 %2").arg(error, response).trimmed());
    if (ok)
    {
        updateSpeedTableCache(axis, speedTable, startSpeedPps, topSpeedPps,
            accelTime10ms, decelTime10ms, accelPattern);
    }
}

void KohzuController::configureSpeedTableAndMoveAbsolute(int axis, qint64 pulse, int speedTable,
    int startSpeedPps, int topSpeedPps, int accelTime10ms, int decelTime10ms,
    int accelPattern, const QString& speedLabel, const QString& moveLabel)
{
    if (!isConnected())
    {
        emit commandFinished(speedLabel, false, QStringLiteral("Kohzu is not connected."));
        emit axisActiveChanged(axis, false);
        return;
    }

    // A lead move may have been acknowledged before the drive has released
    // its previous profile.  Do not submit WTB/APS until the controller
    // explicitly reports the axis idle; otherwise ARIES returns 605/803 and
    // the caller may mistake the rejected setup for a motor failure.
    QString idleStatus;
    if (!waitForAxisIdle(axis, 5000, &idleStatus))
    {
        const QString message = QStringLiteral("axis %1 not idle before WTB; move not sent (status=%2)")
            .arg(axis).arg(idleStatus);
        emit commandFinished(speedLabel, false, message);
        emit commandFinished(moveLabel, false, QStringLiteral("speed table setup blocked: %1").arg(message));
        emit axisActiveChanged(axis, false);
        return;
    }

    QString error;
    if (!speedTableCacheMatches(axis, speedTable, startSpeedPps, topSpeedPps,
        accelTime10ms, decelTime10ms, accelPattern))
    {
        const QString speedResponse = writeSpeedTable(axis, speedTable, startSpeedPps, topSpeedPps,
            accelTime10ms, decelTime10ms, accelPattern, &error);
        const bool speedOk = !speedResponse.isEmpty() && speedResponse.startsWith(QLatin1Char('C'));
        emit commandFinished(speedLabel, speedOk, speedOk ? speedResponse : QStringLiteral("%1 %2").arg(error, speedResponse).trimmed());
        if (!speedOk)
        {
            emit commandFinished(moveLabel, false, QStringLiteral("speed table setup failed; move not sent"));
            emit axisActiveChanged(axis, false);
            return;
        }
        updateSpeedTableCache(axis, speedTable, startSpeedPps, topSpeedPps,
            accelTime10ms, decelTime10ms, accelPattern);
    }

    const QString moveCommand = QStringLiteral("APS%1/%2/%3/1").arg(axis).arg(speedTable).arg(pulse);
    error.clear();
    const QString moveResponse = query(moveCommand, &error);
    const bool moveOk = !moveResponse.isEmpty() && moveResponse.startsWith(QLatin1Char('C'));
    emit commandFinished(moveLabel, moveOk, moveOk ? moveResponse : QStringLiteral("%1 %2").arg(error, moveResponse).trimmed());
    if (moveOk)
    {
        const bool hadKnownPosition = lastKnownPosition_.contains(axis);
        const qint64 previousPulse = lastKnownPosition_.value(axis, pulse);
        const qint64 releaseAfterMs = hadKnownPosition
            ? estimateMoveReleaseMs(pulse - previousPulse, topSpeedPps, accelTime10ms, decelTime10ms)
            : kDefaultActiveReleaseMs;
        lastKnownPosition_[axis] = pulse;
        emit axisUpdated(axis, pulse, QStringLiteral("Estimated after APS; waiting for RDP refresh."));
        markAxisActive(axis, pulse, true, releaseAfterMs);
    }
    else
    {
        emit axisActiveChanged(axis, false);
    }
}

void KohzuController::recoverAfterInterruptedSession()
{
    if (!isConnected())
    {
        emit logMessage(QStringLiteral("Kohzu recovery skipped: controller is not connected."));
        return;
    }

    emit logMessage(QStringLiteral("Kohzu recovery: issuing controlled stop after interrupted CT session."));
    const bool stopOk = [&]() {
        activePollTimer_.stop();
        constexpr int kStopAttempts = 2;
        constexpr int kStopTimeoutMs = 4000;
        constexpr int kStopRetryDelayMs = 250;
        QString error;
        QString response;
        bool ok = false;
        for (int attempt = 0; attempt < kStopAttempts; ++attempt)
        {
            error.clear();
            response = query(QStringLiteral("STP0/0"), &error, kStopTimeoutMs);
            ok = !response.isEmpty() && response.startsWith(QLatin1Char('C'));
            if (ok)
            {
                break;
            }
            if (attempt + 1 < kStopAttempts)
            {
                emit logMessage(QStringLiteral("Kohzu recovery stop retry %1/%2 after timeout/error: %3")
                    .arg(attempt + 1)
                    .arg(kStopAttempts - 1)
                    .arg(error.isEmpty() ? response : error));
                QThread::msleep(kStopRetryDelayMs);
            }
        }
        emit commandFinished(QStringLiteral("Kohzu stop all"), ok,
            ok ? response : QStringLiteral("%1 %2").arg(error, response).trimmed());
        activeAxes_.clear();
        lastActivePosition_.clear();
        activeTargetPosition_.clear();
        activeStartedMs_.clear();
        activeReleaseAfterMs_.clear();
        activeStableCount_.clear();
        activeReadMissCount_.clear();
        activeLastMotionState_.clear();
        speedTableCache_.clear();
        activePollTimer_.stop();
        return ok;
    }();
    bool allIdle = stopOk;
    for (int axis : axes_)
    {
        QString status;
        if (!waitForAxisIdle(axis, 8000, &status))
        {
            allIdle = false;
            emit logMessage(QStringLiteral("Kohzu recovery incomplete: axis %1 is not confirmed idle: %2")
                .arg(axis).arg(status));
            continue;
        }
        emit axisUpdated(axis, lastKnownPosition_.value(axis, 0), QStringLiteral("Recovery idle"));
    }
    speedTableCache_.clear();
    emit commandFinished(QStringLiteral("Kohzu recovery"), allIdle,
        allIdle ? QStringLiteral("all configured axes confirmed idle")
                : QStringLiteral("manual confirmation required; one or more axes are not confirmed idle"));
    emit logMessage(allIdle
        ? QStringLiteral("Kohzu recovery completed; speed-table cache cleared.")
        : QStringLiteral("Kohzu recovery blocked; controller state is not fully confirmed idle."));
}

void KohzuController::configureSpeedTableAndMoveRelative(int axis, qint64 pulse, int speedTable,
    int startSpeedPps, int topSpeedPps, int accelTime10ms, int decelTime10ms,
    int accelPattern, const QString& speedLabel, const QString& moveLabel)
{
    if (!isConnected())
    {
        emit commandFinished(speedLabel, false, QStringLiteral("Kohzu is not connected."));
        emit axisActiveChanged(axis, false);
        return;
    }

    QString idleStatus;
    if (!waitForAxisIdle(axis, 5000, &idleStatus))
    {
        const QString message = QStringLiteral("axis %1 not idle before relative WTB/APS: %2").arg(axis).arg(idleStatus);
        emit commandFinished(speedLabel, false, message);
        emit commandFinished(moveLabel, false, QStringLiteral("speed table setup blocked: %1").arg(message));
        emit axisActiveChanged(axis, false);
        return;
    }

    QString error;
    if (!speedTableCacheMatches(axis, speedTable, startSpeedPps, topSpeedPps,
        accelTime10ms, decelTime10ms, accelPattern))
    {
        const QString speedResponse = writeSpeedTable(axis, speedTable, startSpeedPps, topSpeedPps,
            accelTime10ms, decelTime10ms, accelPattern, &error);
        const bool speedOk = !speedResponse.isEmpty() && speedResponse.startsWith(QLatin1Char('C'));
        emit commandFinished(speedLabel, speedOk, speedOk ? speedResponse : QStringLiteral("%1 %2").arg(error, speedResponse).trimmed());
        if (!speedOk)
        {
            emit commandFinished(moveLabel, false, QStringLiteral("speed table setup failed; move not sent"));
            emit axisActiveChanged(axis, false);
            return;
        }
        updateSpeedTableCache(axis, speedTable, startSpeedPps, topSpeedPps,
            accelTime10ms, decelTime10ms, accelPattern);
    }

    const QString moveCommand = QStringLiteral("RPS%1/%2/%3/1").arg(axis).arg(speedTable).arg(pulse);
    error.clear();
    const QString moveResponse = query(moveCommand, &error);
    const bool moveOk = !moveResponse.isEmpty() && moveResponse.startsWith(QLatin1Char('C'));
    emit commandFinished(moveLabel, moveOk, moveOk ? moveResponse : QStringLiteral("%1 %2").arg(error, moveResponse).trimmed());
    if (moveOk)
    {
        const bool hasKnownPosition = lastKnownPosition_.contains(axis);
        const qint64 targetPulse = hasKnownPosition ? lastKnownPosition_.value(axis) + pulse : 0;
        const qint64 releaseAfterMs = hasKnownPosition
            ? estimateMoveReleaseMs(pulse, topSpeedPps, accelTime10ms, decelTime10ms)
            : kDefaultActiveReleaseMs;
        if (hasKnownPosition)
        {
            lastKnownPosition_[axis] = targetPulse;
            emit axisUpdated(axis, targetPulse, QStringLiteral("Estimated after RPS; waiting for RDP refresh."));
        }
        markAxisActive(axis, targetPulse, hasKnownPosition, releaseAfterMs);
    }
    else
    {
        emit axisActiveChanged(axis, false);
    }
}

void KohzuController::stopAxis(int axis, const QString& label)
{
    if (!isConnected())
    {
        emit commandFinished(label, false, QStringLiteral("Kohzu is not connected."));
        return;
    }

    QString error;
    const QString response = query(QStringLiteral("STP%1/0").arg(axis), &error);
    const bool ok = !response.isEmpty() && response.startsWith(QLatin1Char('C'));
    emit commandFinished(label, ok, ok ? response : QStringLiteral("%1 %2").arg(error, response).trimmed());

    activeAxes_.remove(axis);
    lastActivePosition_.remove(axis);
    activeTargetPosition_.remove(axis);
    activeStartedMs_.remove(axis);
    activeReleaseAfterMs_.remove(axis);
    activeStableCount_.remove(axis);
    activeReadMissCount_.remove(axis);
    emit axisActiveChanged(axis, false);
    if (activeAxes_.isEmpty())
    {
        activePollTimer_.stop();
    }
}

void KohzuController::stopAllAxes()
{
    if (!isConnected())
    {
        emit commandFinished(QStringLiteral("Kohzu stop all"), false, QStringLiteral("Kohzu is not connected."));
        return;
    }

    // Do not let the active-position poller compete with the stop transaction.
    // During a continuous fly move the controller can also take longer than the
    // normal command timeout to acknowledge STP0/0 while decelerating.  Use a
    // bounded, stop-specific retry window; never report STOPPED unless an ACK
    // was actually received.
    activePollTimer_.stop();
    constexpr int kStopAttempts = 2;
    constexpr int kStopTimeoutMs = 4000;
    constexpr int kStopRetryDelayMs = 250;
    QString error;
    QString response;
    bool ok = false;
    for (int attempt = 0; attempt < kStopAttempts; ++attempt)
    {
        error.clear();
        response = query(QStringLiteral("STP0/0"), &error, kStopTimeoutMs);
        ok = !response.isEmpty() && response.startsWith(QLatin1Char('C'));
        if (ok)
        {
            break;
        }
        if (attempt + 1 < kStopAttempts)
        {
            emit logMessage(QStringLiteral("Kohzu stop all retry %1/%2 after timeout/error: %3")
                .arg(attempt + 1)
                .arg(kStopAttempts - 1)
                .arg(error.isEmpty() ? response : error));
            QThread::msleep(kStopRetryDelayMs);
        }
    }
    emit commandFinished(QStringLiteral("Kohzu stop all"), ok,
        ok ? response : QStringLiteral("%1 %2").arg(error, response).trimmed());

    const QList<int> activeAxes = activeAxes_.values();
    activeAxes_.clear();
    lastActivePosition_.clear();
    activeTargetPosition_.clear();
    activeStartedMs_.clear();
    activeReleaseAfterMs_.clear();
    activeStableCount_.clear();
    activeReadMissCount_.clear();
    activeLastMotionState_.clear();
    speedTableCache_.clear();
    activePollTimer_.stop();
    for (int axis : activeAxes)
    {
        emit axisActiveChanged(axis, false);
    }
}

void KohzuController::pollAllAxes()
{
    if (!idlePollingEnabled_)
    {
        return;
    }
    pollAllAxesWithTimeout(kPollAxisTimeoutMs);
}

void KohzuController::pollAllAxesWithTimeout(int timeoutMs)
{
    if (!isConnected())
    {
        return;
    }

    QString rawResponse;
    const QHash<int, qint64> positions = readAxisPositions(toAxisList(axes_), timeoutMs, &rawResponse);

    int updatedCount = 0;
    for (int axis : axes_)
    {
        if (!positions.contains(axis))
        {
            continue;
        }
        const qint64 position = positions.value(axis);
        lastKnownPosition_[axis] = position;
        emit axisUpdated(axis, position, QStringLiteral("RDP"));
        if (activeAxes_.contains(axis))
        {
            updateActiveAxisState(axis, position, QStringLiteral("RDP"));
        }
        ++updatedCount;
    }

    if (timeoutMs >= kInitialAxisTimeoutMs)
    {
        QStringList missingAxes;
        for (int axis : axes_)
        {
            if (!positions.contains(axis))
            {
                missingAxes.push_back(QString::number(axis));
            }
        }
        const QString missingText = missingAxes.isEmpty()
            ? QStringLiteral("")
            : QStringLiteral(" Missing axes: %1.").arg(missingAxes.join(QStringLiteral(",")));
        emit logMessage(QStringLiteral("Kohzu initial position refresh: %1/%2 axes updated.%3")
            .arg(updatedCount)
            .arg(axes_.size())
            .arg(missingText));
    }
    else if (updatedCount == 0)
    {
        const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
        const bool busy = isKohzuBusyError(rawResponse);
        if (nowMs - lastPollFailureLogMs_ >= 10000)
        {
            lastPollFailureLogMs_ = nowMs;
            emit logMessage(QStringLiteral("Kohzu position refresh failed: %1 Raw: %2")
                .arg(busy
                    ? QStringLiteral("controller busy/possibly occupied by another client (803).")
                    : QStringLiteral("no RDP responses received."))
                .arg(rawResponse.left(300)));
        }
    }
}

void KohzuController::pollActiveAxes()
{
    if (!isConnected())
    {
        activePollTimer_.stop();
        return;
    }

    const QList<int> activeAxes = activeAxes_.values();
    for (int axis : activeAxes)
    {
        qint64 position = 0;
        QString status;
        QString readError;
        // RDP alone cannot tell whether the controller has released the
        // previous motion.  Read STR in the same poll so a completed move can
        // be released immediately instead of waiting the 15 s estimate.
        const bool positionRead = readAxis(axis, &position, &status, &readError,
            kPollAxisTimeoutMs, true);
        if (!positionRead)
        {
            const int missCount = activeReadMissCount_.value(axis, 0) + 1;
            activeReadMissCount_[axis] = missCount;
            if (missCount == 1 || missCount % 10 == 0)
            {
                emit logMessage(QStringLiteral("Axis %1 active read missed %2 time(s); waiting for next RDP refresh.")
                    .arg(axis)
                    .arg(missCount));
            }
            const qint64 startedMs = activeStartedMs_.value(axis, QDateTime::currentMSecsSinceEpoch());
            const qint64 releaseAfterMs = activeReleaseAfterMs_.value(axis, kDefaultActiveReleaseMs);
            if (QDateTime::currentMSecsSinceEpoch() - startedMs >= releaseAfterMs)
            {
                activeAxes_.remove(axis);
                lastActivePosition_.remove(axis);
                activeTargetPosition_.remove(axis);
                activeStartedMs_.remove(axis);
                activeReleaseAfterMs_.remove(axis);
                activeStableCount_.remove(axis);
                activeReadMissCount_.remove(axis);
                emit axisActiveChanged(axis, false);
                emit logMessage(QStringLiteral("Axis %1 position/status reads are unavailable; releasing UI after estimated move time %2 ms.")
                    .arg(axis)
                    .arg(releaseAfterMs));
            }
            continue;
        }
        activeReadMissCount_.remove(axis);
        emit axisUpdated(axis, position, status);
        const AxisMotionState decodedState = parseAxisMotionState(status);
        const int decodedValue = static_cast<int>(decodedState);
        const int previousValue = activeLastMotionState_.value(axis, -1);
        if (decodedValue != previousValue)
        {
            activeLastMotionState_[axis] = decodedValue;
            if (decodedState == AxisMotionState::Moving)
            {
                emit logMessage(QStringLiteral("Axis %1 STR confirms moving; position=%2.")
                    .arg(axis).arg(position));
            }
            else if (decodedState == AxisMotionState::Idle)
            {
                emit logMessage(QStringLiteral("Axis %1 STR confirms idle; releasing active state.")
                    .arg(axis));
            }
            else
            {
                emit logMessage(QStringLiteral("Axis %1 STR unavailable/unrecognized; using estimated release timeout. Raw: %2")
                    .arg(axis).arg(status.left(220)));
            }
        }
        updateActiveAxisState(axis, position, status);
    }

    if (activeAxes_.isEmpty())
    {
        activePollTimer_.stop();
    }
}

void KohzuController::setIdlePollingEnabled(bool enabled)
{
    idlePollingEnabled_ = enabled;
    if (!enabled)
    {
        idlePollTimer_.stop();
        emit logMessage(QStringLiteral("Kohzu idle all-axis polling paused."));
        return;
    }
    if (isConnected() && !idlePollTimer_.isActive())
    {
        idlePollTimer_.start();
    }
    emit logMessage(QStringLiteral("Kohzu idle all-axis polling resumed."));
}

bool KohzuController::isConnected() const
{
    return socket_ && socket_->state() == QAbstractSocket::ConnectedState;
}

QString KohzuController::query(const QString& command, QString* error, int timeoutMs)
{
    if (!isConnected())
    {
        if (error)
        {
            *error = QStringLiteral("Kohzu controller is not connected.");
        }
        return {};
    }

    while (socket_->bytesAvailable() > 0)
    {
        socket_->readAll();
    }

    const quint64 sequence = ++commandSequence_;
    const qint64 sentAtMs = QDateTime::currentMSecsSinceEpoch();
    const bool isRdpQuery = command.trimmed().startsWith(QStringLiteral("RDP"), Qt::CaseInsensitive);
    const bool verboseRdpTrace = qEnvironmentVariableIntValue("CT_VERBOSE_KOHZU_RDP_LOG") != 0;
    const bool traceRdp = !isRdpQuery || verboseRdpTrace
        || lastRdpTraceLogMs_ <= 0
        || sentAtMs - lastRdpTraceLogMs_ >= kRdpTraceIntervalMs;
    if (isRdpQuery && !traceRdp)
    {
        ++suppressedRdpTraceCount_;
    }
    if (traceRdp)
    {
        if (isRdpQuery && suppressedRdpTraceCount_ > 0)
        {
            emit logMessage(QStringLiteral("Kohzu RDP polling trace: %1 routine command traces suppressed; set CT_VERBOSE_KOHZU_RDP_LOG=1 for full detail.")
                .arg(suppressedRdpTraceCount_));
            suppressedRdpTraceCount_ = 0;
        }
        emit logMessage(QStringLiteral("Kohzu command #%1 send t=%2 thread=%3: %4")
            .arg(sequence)
            .arg(sentAtMs)
            .arg(reinterpret_cast<quintptr>(QThread::currentThreadId()), 0, 16)
            .arg(command));
        if (isRdpQuery)
        {
            lastRdpTraceLogMs_ = sentAtMs;
        }
    }
    const QByteArray payload = command.toLatin1() + "\r\n";
    if (socket_->write(payload) != payload.size() || !socket_->waitForBytesWritten(timeoutMs))
    {
        if (error)
        {
            *error = socket_->errorString();
        }
        return {};
    }

    const QString expectedToken = responseTokenForCommand(command);
    QByteArray response;
    QString lastResponseText;
    const qint64 deadlineMs = QDateTime::currentMSecsSinceEpoch() + timeoutMs;
    while (QDateTime::currentMSecsSinceEpoch() < deadlineMs)
    {
        if (socket_->bytesAvailable() <= 0)
        {
            const int waitMs = static_cast<int>(std::max<qint64>(1, deadlineMs - QDateTime::currentMSecsSinceEpoch()));
            if (!socket_->waitForReadyRead(waitMs))
            {
                break;
            }
        }

        response += socket_->readAll();
        lastResponseText = QString::fromLatin1(response).trimmed();
        const QString matchedResponse = findResponseForToken(lastResponseText, expectedToken);
        if (!matchedResponse.isEmpty())
        {
            if (traceRdp)
            {
                emit logMessage(QStringLiteral("Kohzu command #%1 response dt=%2 ms: %3")
                    .arg(sequence)
                    .arg(QDateTime::currentMSecsSinceEpoch() - sentAtMs)
                    .arg(matchedResponse));
            }
            return matchedResponse;
        }
    }

    if (error)
    {
        *error = lastResponseText.isEmpty()
            ? QStringLiteral("Timed out waiting for %1: %2").arg(expectedToken, socket_->errorString())
            : QStringLiteral("Timed out waiting for %1; last response was: %2").arg(expectedToken, lastResponseText);
    }
    emit logMessage(QStringLiteral("Kohzu command #%1 timeout dt=%2 ms: %3; raw=%4")
        .arg(sequence)
        .arg(QDateTime::currentMSecsSinceEpoch() - sentAtMs)
        .arg(command)
        .arg(lastResponseText.left(300)));
    return {};
}

bool KohzuController::waitForAxisIdle(int axis, int timeoutMs, QString* statusRaw)
{
    const qint64 deadline = QDateTime::currentMSecsSinceEpoch() + std::max(0, timeoutMs);
    QString lastStatus;
    while (QDateTime::currentMSecsSinceEpoch() <= deadline)
    {
        QString error;
        const QString response = query(QStringLiteral("STR%1").arg(axis), &error, 700);
        lastStatus = response.isEmpty() ? error : response;
        const AxisMotionState state = parseAxisMotionState(response);
        if (state == AxisMotionState::Idle)
        {
            if (statusRaw) { *statusRaw = response; }
            emit logMessage(QStringLiteral("Kohzu axis %1 confirmed idle before motion command: %2").arg(axis).arg(response));
            return true;
        }
        if (state == AxisMotionState::Unknown)
        {
            emit logMessage(QStringLiteral("Kohzu axis %1 status not understood while waiting for idle: %2").arg(axis).arg(lastStatus));
        }
        QThread::msleep(100);
    }
    if (statusRaw) { *statusRaw = lastStatus; }
    emit logMessage(QStringLiteral("Kohzu axis %1 did not confirm idle within %2 ms: %3").arg(axis).arg(timeoutMs).arg(lastStatus));
    return false;
}

QString KohzuController::writeSpeedTable(int axis, int speedTable, int startSpeedPps,
    int topSpeedPps, int accelTime10ms, int decelTime10ms, int accelPattern, QString* error)
{
    const QString command = QStringLiteral("WTB%1/%2/%3/%4/%5/%6/%7")
        .arg(axis)
        .arg(speedTable)
        .arg(startSpeedPps)
        .arg(topSpeedPps)
        .arg(accelTime10ms)
        .arg(decelTime10ms)
        .arg(accelPattern);

    // A fly scan follows a lead move immediately. ARIES can acknowledge that
    // move before it accepts WTB, so retry only the transient controller codes.
    constexpr int kAttempts = 4;
    constexpr int kRetryDelayMs[] = {120, 240, 480};
    QString localError;
    for (int attempt = 0; attempt < kAttempts; ++attempt)
    {
        localError.clear();
        const QString response = query(command, &localError);
        if (!response.isEmpty() && !isTransientSpeedTableError(response))
        {
            return response;
        }
        if (attempt + 1 >= kAttempts)
        {
            if (error)
            {
                *error = localError;
            }
            return response;
        }
        emit logMessage(QStringLiteral("Kohzu %1 retry %2/%3 after transient response: %4")
            .arg(command)
            .arg(attempt + 1)
            .arg(kAttempts - 1)
            .arg(response.isEmpty() ? localError : response));
        QThread::msleep(kRetryDelayMs[attempt]);
    }
    return {};
}

bool KohzuController::speedTableCacheMatches(int axis, int speedTable, int startSpeedPps, int topSpeedPps,
    int accelTime10ms, int decelTime10ms, int accelPattern) const
{
    const auto it = speedTableCache_.constFind(axis);
    if (it == speedTableCache_.constEnd())
    {
        return false;
    }
    return it->speedTable == speedTable
        && it->startSpeedPps == startSpeedPps
        && it->topSpeedPps == topSpeedPps
        && it->accelTime10ms == accelTime10ms
        && it->decelTime10ms == decelTime10ms
        && it->accelPattern == accelPattern;
}

void KohzuController::updateSpeedTableCache(int axis, int speedTable, int startSpeedPps, int topSpeedPps,
    int accelTime10ms, int decelTime10ms, int accelPattern)
{
    speedTableCache_[axis] = SpeedTableCache{
        speedTable,
        startSpeedPps,
        topSpeedPps,
        accelTime10ms,
        decelTime10ms,
        accelPattern,
    };
}

QHash<int, qint64> KohzuController::readAxisPositions(const QList<int>& axisList, int timeoutMs, QString* rawResponse)
{
    QHash<int, qint64> positions;
    if (!isConnected() || axisList.isEmpty())
    {
        if (rawResponse)
        {
            *rawResponse = {};
        }
        return positions;
    }

    static const QRegularExpression rdpPattern(
        QStringLiteral("\\bC\\s+RDP(\\d+)\\s+(-?\\d+)\\b"),
        QRegularExpression::CaseInsensitiveOption);
    QStringList rawLines;
    for (int axis : axisList)
    {
        QString error;
        QString response;
        for (int attempt = 0; attempt <= kRdpRetryCount; ++attempt)
        {
            if (attempt > 0)
            {
                QThread::msleep(30 * attempt);
            }
            response = query(QStringLiteral("RDP%1").arg(axis), &error, timeoutMs);
            if (!response.isEmpty() && !isKohzuBusyError(response))
            {
                break;
            }
        }

        if (response.isEmpty())
        {
            rawLines.push_back(error.isEmpty()
                ? QStringLiteral("RDP%1: no response").arg(axis)
                : QStringLiteral("RDP%1: %2").arg(axis).arg(error));
            continue;
        }

        rawLines.push_back(response);
        const QRegularExpressionMatch match = rdpPattern.match(response);
        if (!match.hasMatch())
        {
            continue;
        }
        const int responseAxis = match.captured(1).toInt();
        if (responseAxis != axis)
        {
            rawLines.push_back(QStringLiteral("RDP%1: ignored response for axis %2").arg(axis).arg(responseAxis));
            continue;
        }
        positions[axis] = match.captured(2).toLongLong();
    }

    if (rawResponse)
    {
        *rawResponse = rawLines.join(QStringLiteral("\n")).trimmed();
    }
    for (auto it = positions.constBegin(); it != positions.constEnd(); ++it)
    {
        lastKnownPosition_[it.key()] = it.value();
    }
    return positions;
}

bool KohzuController::readAxis(int axis, qint64* positionPulse, QString* statusRaw, QString* error, int timeoutMs, bool readStatus)
{
    QString localError;
    QString rawResponse;
    const QHash<int, qint64> positions = readAxisPositions(QList<int>{axis}, timeoutMs, &rawResponse);
    if (!positions.contains(axis))
    {
        if (error)
        {
            *error = rawResponse.isEmpty()
                ? QStringLiteral("Timed out waiting for RDP%1: %2").arg(axis).arg(localError)
                : QStringLiteral("No matching RDP%1 response; raw response was: %2").arg(axis).arg(rawResponse);
        }
        return false;
    }

    const qint64 parsedPosition = positions.value(axis);

    if (positionPulse)
    {
        *positionPulse = parsedPosition;
    }
    if (statusRaw)
    {
        *statusRaw = QStringLiteral("RDP only");
    }
    lastKnownPosition_[axis] = parsedPosition;

    if (!readStatus)
    {
        if (statusRaw)
        {
            *statusRaw = QStringLiteral("RDP only");
        }
        return true;
    }

    const int statusTimeoutMs = std::max(150, std::min(timeoutMs / 2, 300));
    const QString statusResponse = query(QStringLiteral("STR%1").arg(axis), &localError, statusTimeoutMs);
    if (statusRaw)
    {
        *statusRaw = statusResponse.isEmpty()
            ? QStringLiteral("STR read unavailable: %1").arg(localError)
            : statusResponse;
    }
    return true;
}

void KohzuController::markAxisActive(int axis, qint64 targetPulse, bool hasTarget, qint64 releaseAfterMs)
{
    const bool wasActive = activeAxes_.contains(axis);
    activeAxes_.insert(axis);
    activeStableCount_[axis] = 0;
    activeReadMissCount_[axis] = 0;
    activeLastMotionState_.remove(axis);
    activeStartedMs_[axis] = QDateTime::currentMSecsSinceEpoch();
    lastActivePosition_.remove(axis);
    if (hasTarget)
    {
        activeTargetPosition_[axis] = targetPulse;
    }
    else
    {
        activeTargetPosition_.remove(axis);
    }
    if (releaseAfterMs > 0)
    {
        activeReleaseAfterMs_[axis] = releaseAfterMs;
    }
    else
    {
        activeReleaseAfterMs_[axis] = kDefaultActiveReleaseMs;
    }
    if (!wasActive)
    {
        emit axisActiveChanged(axis, true);
    }
    if (!activePollTimer_.isActive())
    {
        activePollTimer_.start();
    }
}

void KohzuController::updateActiveAxisState(int axis, qint64 positionPulse, const QString& statusRaw)
{
    auto finishAxis = [&]() {
        activeAxes_.remove(axis);
        lastActivePosition_.remove(axis);
        activeTargetPosition_.remove(axis);
        activeStartedMs_.remove(axis);
        activeReleaseAfterMs_.remove(axis);
        activeStableCount_.remove(axis);
        activeLastMotionState_.remove(axis);
        activeReadMissCount_.remove(axis);
        emit axisActiveChanged(axis, false);
    };

    const AxisMotionState motionState = parseAxisMotionState(statusRaw);
    if (motionState == AxisMotionState::Moving)
    {
        lastActivePosition_[axis] = positionPulse;
        activeStableCount_[axis] = 0;
        return;
    }
    if (motionState == AxisMotionState::Idle)
    {
        finishAxis();
        return;
    }

    if (!lastActivePosition_.contains(axis) || lastActivePosition_.value(axis) != positionPulse)
    {
        lastActivePosition_[axis] = positionPulse;
        activeStableCount_[axis] = 0;
        return;
    }

    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
    const qint64 startedMs = activeStartedMs_.value(axis, nowMs);
    // APS/RPS acknowledgement and an RDP target position can arrive while
    // the axis is still decelerating.  Reconfiguring WTB at that instant is
    // rejected by ARIES with 605.  Wait out the conservative motion estimate
    // before treating position stability as a completed move.
    const qint64 releaseAfterMs = activeReleaseAfterMs_.value(axis, kDefaultActiveReleaseMs);
    if (nowMs - startedMs < std::max(kMinPositionStableMs, releaseAfterMs))
    {
        return;
    }

    const int stableCount = activeStableCount_.value(axis, 0) + 1;
    activeStableCount_[axis] = stableCount;
    if (stableCount >= kStablePollsBeforeInactive)
    {
        finishAxis();
    }
}

} // namespace ct
