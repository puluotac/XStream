#include "hardware/kohzu/KohzuController.h"

#include <QRegularExpression>

namespace ct
{

namespace
{

constexpr int kIdlePollMs = 1500;
constexpr int kActivePollMs = 80;
constexpr int kStablePollsBeforeInactive = 6;

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
    if (!socket_->waitForConnected(1500))
    {
        const QString error = socket_->errorString();
        delete socket_;
        socket_ = nullptr;
        emit connectionChanged(false);
        emit logMessage(QStringLiteral("Kohzu connect failed: %1").arg(error));
        return;
    }

    emit connectionChanged(true);
    emit logMessage(QStringLiteral("Kohzu connected: %1:%2").arg(host).arg(port));

    QString idnError;
    const QString idn = query(QStringLiteral("IDN"), &idnError);
    if (!idn.isEmpty())
    {
        emit logMessage(QStringLiteral("Kohzu IDN: %1").arg(idn));
    }
    else if (!idnError.isEmpty())
    {
        emit logMessage(QStringLiteral("Kohzu IDN failed: %1").arg(idnError));
    }

    pollAllAxes();
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

void KohzuController::pollAllAxes()
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
        if (!readAxis(axis, &position, &status, &error))
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

bool KohzuController::readAxis(int axis, qint64* positionPulse, QString* statusRaw, QString* error)
{
    QString localError;
    const QString posResponse = query(QStringLiteral("RDP%1").arg(axis), &localError, 700);
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

    const QString statusResponse = query(QStringLiteral("STR%1").arg(axis), &localError, 700);
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
