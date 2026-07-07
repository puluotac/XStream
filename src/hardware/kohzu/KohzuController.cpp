#include "hardware/kohzu/KohzuController.h"

#include <QDateTime>
#include <QRegularExpression>
#include <QStringList>
#include <QThread>

namespace ct
{

namespace
{

constexpr int kIdlePollMs = 1500;
constexpr int kActivePollMs = 150;
constexpr int kStablePollsBeforeInactive = 2;
constexpr qint64 kMinPositionStableMs = 160;
constexpr int kConnectTimeoutMs = 1500;
constexpr int kHandshakeTimeoutMs = 2500;
constexpr int kInitialAxisTimeoutMs = 2000;
constexpr int kPollAxisTimeoutMs = 700;

enum class AxisMotionState
{
    Unknown,
    Moving,
    Idle
};

AxisMotionState parseAxisMotionState(const QString& statusRaw)
{
    // ARIES/LYNX manual: normal status response is "C STR a b c d e f g".
    // Field b is the driving state: 0 stopped, 1 operating, 2 feedback operating.
    static const QRegularExpression statusPattern(
        QStringLiteral("\\bC\\s+STR\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\s+(-?\\d+)\\b"),
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
    activeStableCount_.clear();
    activeReadMissCount_.clear();
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
            emit logMessage(QStringLiteral("Kohzu controller verified by RDP batch: %1/%2 axes responded.")
                .arg(positions.size())
                .arg(axes_.size()));
            for (auto it = positions.constBegin(); it != positions.constEnd(); ++it)
            {
                emit axisUpdated(it.key(), it.value(), QStringLiteral("RDP batch"));
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
    activeStableCount_.clear();
    activeReadMissCount_.clear();
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

    const QString command = QStringLiteral("WTB%1/%2/%3/%4/%5/%6/%7")
        .arg(axis)
        .arg(speedTable)
        .arg(startSpeedPps)
        .arg(topSpeedPps)
        .arg(accelTime10ms)
        .arg(decelTime10ms)
        .arg(accelPattern);
    QString error;
    const QString response = query(command, &error);
    const bool ok = !response.isEmpty() && response.startsWith(QLatin1Char('C'));
    emit commandFinished(label, ok, ok ? response : QStringLiteral("%1 %2").arg(error, response).trimmed());
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

    const QString speedCommand = QStringLiteral("WTB%1/%2/%3/%4/%5/%6/%7")
        .arg(axis)
        .arg(speedTable)
        .arg(startSpeedPps)
        .arg(topSpeedPps)
        .arg(accelTime10ms)
        .arg(decelTime10ms)
        .arg(accelPattern);
    QString error;
    const QString speedResponse = query(speedCommand, &error);
    const bool speedOk = !speedResponse.isEmpty() && speedResponse.startsWith(QLatin1Char('C'));
    emit commandFinished(speedLabel, speedOk, speedOk ? speedResponse : QStringLiteral("%1 %2").arg(error, speedResponse).trimmed());
    if (!speedOk)
    {
        emit commandFinished(moveLabel, false, QStringLiteral("speed table setup failed; move not sent"));
        emit axisActiveChanged(axis, false);
        return;
    }

    const QString moveCommand = QStringLiteral("APS%1/%2/%3/1").arg(axis).arg(speedTable).arg(pulse);
    error.clear();
    const QString moveResponse = query(moveCommand, &error);
    const bool moveOk = !moveResponse.isEmpty() && moveResponse.startsWith(QLatin1Char('C'));
    emit commandFinished(moveLabel, moveOk, moveOk ? moveResponse : QStringLiteral("%1 %2").arg(error, moveResponse).trimmed());
    if (moveOk)
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

    const QString speedCommand = QStringLiteral("WTB%1/%2/%3/%4/%5/%6/%7")
        .arg(axis)
        .arg(speedTable)
        .arg(startSpeedPps)
        .arg(topSpeedPps)
        .arg(accelTime10ms)
        .arg(decelTime10ms)
        .arg(accelPattern);
    QString error;
    const QString speedResponse = query(speedCommand, &error);
    const bool speedOk = !speedResponse.isEmpty() && speedResponse.startsWith(QLatin1Char('C'));
    emit commandFinished(speedLabel, speedOk, speedOk ? speedResponse : QStringLiteral("%1 %2").arg(error, speedResponse).trimmed());
    if (!speedOk)
    {
        emit commandFinished(moveLabel, false, QStringLiteral("speed table setup failed; move not sent"));
        emit axisActiveChanged(axis, false);
        return;
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

    QString error;
    const QString response = query(QStringLiteral("STP0/0"), &error);
    const bool ok = !response.isEmpty() && response.startsWith(QLatin1Char('C'));
    emit commandFinished(QStringLiteral("Kohzu stop all"), ok, ok ? response : QStringLiteral("%1 %2").arg(error, response).trimmed());

    const QList<int> activeAxes = activeAxes_.values();
    activeAxes_.clear();
    lastActivePosition_.clear();
    activeTargetPosition_.clear();
    activeStartedMs_.clear();
    activeStableCount_.clear();
    activeReadMissCount_.clear();
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
        emit axisUpdated(axis, position, QStringLiteral("RDP batch"));
        if (activeAxes_.contains(axis))
        {
            updateActiveAxisState(axis, position, QStringLiteral("RDP batch"));
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
        emit logMessage(QStringLiteral("Kohzu position refresh failed: no RDP responses received. Raw: %1")
            .arg(rawResponse.left(300)));
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
    const QHash<int, qint64> positions = readAxisPositions(activeAxes, kPollAxisTimeoutMs);
    for (int axis : activeAxes)
    {
        if (!positions.contains(axis))
        {
            const int missCount = activeReadMissCount_.value(axis, 0) + 1;
            activeReadMissCount_[axis] = missCount;
            if (missCount == 1 || missCount % 10 == 0)
            {
                emit logMessage(QStringLiteral("Axis %1 active read missed %2 time(s); waiting for next RDP refresh.")
                    .arg(axis)
                    .arg(missCount));
            }
            continue;
        }
        const qint64 position = positions.value(axis);
        const QString status = QStringLiteral("RDP active batch");
        activeReadMissCount_.remove(axis);
        emit axisUpdated(axis, position, status);
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
            return matchedResponse;
        }
    }

    if (error)
    {
        *error = lastResponseText.isEmpty()
            ? QStringLiteral("Timed out waiting for %1: %2").arg(expectedToken, socket_->errorString())
            : QStringLiteral("Timed out waiting for %1; last response was: %2").arg(expectedToken, lastResponseText);
    }
    return {};
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

    QSet<int> axisSet;
    for (int axis : axisList)
    {
        axisSet.insert(axis);
    }

    for (int axis : axisList)
    {
        const QByteArray payload = QStringLiteral("RDP%1\r\n").arg(axis).toLatin1();
        if (socket_->write(payload) != payload.size() || !socket_->waitForBytesWritten(timeoutMs))
        {
            emit logMessage(QStringLiteral("Axis %1 read request failed: %2").arg(axis).arg(socket_->errorString()));
            if (rawResponse)
            {
                *rawResponse = {};
            }
            return positions;
        }
        QThread::msleep(2);
    }

    QByteArray responseBytes;
    static const QRegularExpression rdpPattern(
        QStringLiteral("\\bC\\s+RDP(\\d+)\\s+(-?\\d+)\\b"),
        QRegularExpression::CaseInsensitiveOption);
    const qint64 deadlineMs = QDateTime::currentMSecsSinceEpoch()
        + std::max(timeoutMs, 500)
        + static_cast<qint64>(axisSet.size()) * 150;
    while (QDateTime::currentMSecsSinceEpoch() < deadlineMs && positions.size() < axisSet.size())
    {
        if (socket_->bytesAvailable() <= 0)
        {
            const int waitMs = static_cast<int>(std::max<qint64>(1, std::min<qint64>(50, deadlineMs - QDateTime::currentMSecsSinceEpoch())));
            if (!socket_->waitForReadyRead(waitMs))
            {
                continue;
            }
        }

        responseBytes += socket_->readAll();
        const QString responseText = QString::fromLatin1(responseBytes);
        QRegularExpressionMatchIterator matches = rdpPattern.globalMatch(responseText);
        while (matches.hasNext())
        {
            const QRegularExpressionMatch match = matches.next();
            const int axis = match.captured(1).toInt();
            if (axisSet.contains(axis))
            {
                positions[axis] = match.captured(2).toLongLong();
            }
        }
    }

    if (rawResponse)
    {
        *rawResponse = QString::fromLatin1(responseBytes).trimmed();
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

void KohzuController::markAxisActive(int axis, qint64 targetPulse, bool hasTarget)
{
    const bool wasActive = activeAxes_.contains(axis);
    activeAxes_.insert(axis);
    activeStableCount_[axis] = 0;
    activeReadMissCount_[axis] = 0;
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
        activeStableCount_.remove(axis);
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

    const auto targetIt = activeTargetPosition_.find(axis);
    if (targetIt != activeTargetPosition_.end() && std::llabs(positionPulse - targetIt.value()) <= 1)
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

    const qint64 startedMs = activeStartedMs_.value(axis, QDateTime::currentMSecsSinceEpoch());
    if (QDateTime::currentMSecsSinceEpoch() - startedMs < kMinPositionStableMs)
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
