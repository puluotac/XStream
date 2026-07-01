#pragma once

#include <QObject>
#include <QHash>
#include <QSet>
#include <QString>
#include <QTcpSocket>
#include <QTimer>
#include <QVector>

namespace ct
{

class KohzuController : public QObject
{
    Q_OBJECT

public:
    explicit KohzuController(QObject* parent = nullptr);
    ~KohzuController() override = default;

public slots:
    void addAxis(int axis);
    void clearAxes();
    void connectToController(const QString& host, quint16 port);
    void shutdown();
    void jogRelativePulse(int axis, qint64 pulse, int speedTable, const QString& label);
    void moveAbsolutePulse(int axis, qint64 pulse, int speedTable, const QString& label);
    void configureSpeedTable(int axis, int speedTable, int startSpeedPps, int topSpeedPps,
        int accelTime10ms, int decelTime10ms, int accelPattern, const QString& label);
    void configureSpeedTableAndMoveAbsolute(int axis, qint64 pulse, int speedTable,
        int startSpeedPps, int topSpeedPps, int accelTime10ms, int decelTime10ms,
        int accelPattern, const QString& speedLabel, const QString& moveLabel);
    void configureSpeedTableAndMoveRelative(int axis, qint64 pulse, int speedTable,
        int startSpeedPps, int topSpeedPps, int accelTime10ms, int decelTime10ms,
        int accelPattern, const QString& speedLabel, const QString& moveLabel);
    void setIdlePollingEnabled(bool enabled);
    void stopAxis(int axis, const QString& label);
    void stopAllAxes();

signals:
    void connectionChanged(bool connected);
    void axisUpdated(int axis, qint64 positionPulse, const QString& statusRaw);
    void axisActiveChanged(int axis, bool active);
    void logMessage(const QString& text);
    void commandFinished(const QString& label, bool ok, const QString& response);

private slots:
    void pollAllAxes();
    void pollActiveAxes();

private:
    bool isConnected() const;
    void closeConnection();
    QString query(const QString& command, QString* error = nullptr, int timeoutMs = 1500);
    bool readAxis(int axis, qint64* positionPulse, QString* statusRaw, QString* error = nullptr, int timeoutMs = 700);
    void pollAllAxesWithTimeout(int timeoutMs);
    void markAxisActive(int axis, qint64 targetPulse = 0, bool hasTarget = false);
    void updateActiveAxisState(int axis, qint64 positionPulse, const QString& statusRaw);

    QTcpSocket* socket_ = nullptr;
    QVector<int> axes_;
    QSet<int> activeAxes_;
    QHash<int, qint64> lastActivePosition_;
    QHash<int, qint64> lastKnownPosition_;
    QHash<int, qint64> activeTargetPosition_;
    QHash<int, qint64> activeStartedMs_;
    QHash<int, int> activeStableCount_;
    QTimer idlePollTimer_;
    QTimer activePollTimer_;
    bool idlePollingEnabled_ = true;
};

} // namespace ct
