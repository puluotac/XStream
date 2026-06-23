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
    void writeSpeedTable(int axis,
                         int speedTable,
                         int startPps,
                         int maxPps,
                         int accelTime10Ms,
                         int decelTime10Ms,
                         int pattern,
                         const QString& label);
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
    bool readAxis(int axis, qint64* positionPulse, QString* statusRaw, QString* error = nullptr);
    void markAxisActive(int axis);
    void updateActiveAxisState(int axis, qint64 positionPulse);

    QTcpSocket* socket_ = nullptr;
    QVector<int> axes_;
    QSet<int> activeAxes_;
    QHash<int, qint64> lastActivePosition_;
    QHash<int, int> activeStableCount_;
    QTimer idlePollTimer_;
    QTimer activePollTimer_;
};

} // namespace ct
