#include "hardware/kohzu/KohzuController.h"

#include <QRegularExpression>
#include <QThread>

namespace ct
{

namespace
{

constexpr int kIdlePollMs = 1500;
constexpr int kActivePollMs = 80;
constexpr int kStablePollsBeforeInactive = 6;
constexpr int kConnectTimeoutMs = 1500;
constexpr int kHandshakeTimeoutMs = 2500;
constexpr int kInitialAxisTimeoutMs = 2000;
constexpr int kPollAxisTimeoutMs = 700;

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
    activeStableCount_.clear();
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
        for (int axis : axes_)
        {
            qint64 position = 0;
            QString status;
            QString axisError;
            if (readAxis(axis, &position, &status, &axisError, kInitialAxisTimeoutMs))
            {
                emit logMessage(QStringLiteral("Kohzu controller verified by axis %1 read.").arg(axis));
                emit axisUpdated(axis, position, status);
                controllerReady = true;
                break;
            }
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
    idlePollTimer_.start();
}

void KohzuController::closeConnection()
{
    idlePollTimer_.stop();
    activePollTimer_.stop();
    const QList<int> activeAxes = activeAxes_.values();
    activeAxes_.clear();
    lastActivePosition_.clear();
    activeStableCount_.clear();
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
        markAxisActive(axis);
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
        markAxisActive(axis);
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
        markAxisActive(axis);
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
    activeStableCount_.remove(axis);
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
    activeStableCount_.clear();
    activePollTimer_.stop();
    for (int axis : activeAxes)
    {
        emit axisActiveChanged(axis, false);
    }
}

void KohzuController::pollAllAxes()
{
    pollAllAxesWithTimeout(kPollAxisTimeoutMs);
}

void KohzuController::pollAllAxesWithTimeout(int timeoutMs)
{
    if (!isConnected())
    {
        return;
    }

    for (int axis : axes_)
    {
        qint64 position = 0;
        QString status;
        QString error;
        if (!readAxis(axis, &position, &status, &error, timeoutMs))
        {
            emit logMessage(QStringLiteral("Axis %1 read failed: %2").arg(axis).arg(error));
            continue;
        }
        emit axisUpdated(axis, position, status);
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
        QString error;
        if (!readAxis(axis, &position, &status, &error))
        {
            emit logMessage(QStringLiteral("Axis %1 active read failed: %2").arg(axis).arg(error));
            continue;
        }
        emit axisUpdated(axis, position, status);
        updateActiveAxisState(axis, position);
    }

    if (activeAxes_.isEmpty())
    {
        activePollTimer_.stop();
    }
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

    QByteArray response;
    if (!socket_->waitForReadyRead(timeoutMs))
    {
        if (error)
        {
            *error = socket_->errorString();
        }
        return {};
    }
    response += socket_->readAll();
    while (socket_->waitForReadyRead(20))
    {
        response += socket_->readAll();
    }

    return QString::fromLatin1(response).trimmed();
}

bool KohzuController::readAxis(int axis, qint64* positionPulse, QString* statusRaw, QString* error, int timeoutMs)
{
    QString localError;
    const QString posResponse = query(QStringLiteral("RDP%1").arg(axis), &localError, timeoutMs);
    if (posResponse.isEmpty())
    {
        if (error)
        {
            *error = localError;
        }
        return false;
    }

    static const QRegularExpression numberAtEnd(QStringLiteral("(-?\\d+)\\s*$"));
    const QRegularExpressionMatch match = numberAtEnd.match(posResponse);
    if (!match.hasMatch())
    {
        if (error)
        {
            *error = QStringLiteral("Unexpected RDP response: %1").arg(posResponse);
        }
        return false;
    }

    const QString statusResponse = query(QStringLiteral("STR%1").arg(axis), &localError, timeoutMs);
    if (statusResponse.isEmpty())
    {
        if (error)
        {
            *error = localError;
        }
        return false;
    }

    if (positionPulse)
    {
        *positionPulse = match.captured(1).toLongLong();
    }
    if (statusRaw)
    {
        *statusRaw = statusResponse;
    }
    return true;
}

void KohzuController::markAxisActive(int axis)
{
    const bool wasActive = activeAxes_.contains(axis);
    activeAxes_.insert(axis);
    activeStableCount_[axis] = 0;
    lastActivePosition_.remove(axis);
    if (!wasActive)
    {
        emit axisActiveChanged(axis, true);
    }
    if (!activePollTimer_.isActive())
    {
        activePollTimer_.start();
    }
}

void KohzuController::updateActiveAxisState(int axis, qint64 positionPulse)
{
    if (!lastActivePosition_.contains(axis) || lastActivePosition_.value(axis) != positionPulse)
    {
        lastActivePosition_[axis] = positionPulse;
        activeStableCount_[axis] = 0;
        return;
    }

    const int stableCount = activeStableCount_.value(axis, 0) + 1;
    activeStableCount_[axis] = stableCount;
    if (stableCount >= kStablePollsBeforeInactive)
    {
        activeAxes_.remove(axis);
        lastActivePosition_.remove(axis);
        activeStableCount_.remove(axis);
        emit axisActiveChanged(axis, false);
    }
}

} // namespace ct
