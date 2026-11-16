#include "hardware/kohzu/KohzuController.h"
#include <QCoreApplication>
#include <QTcpServer>
#include <QTcpSocket>
#include <QThread>
#include <atomic>
#include <future>
#include <iostream>
#include <thread>

// Loopback only: exercises the production command path without beamline access.
int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    std::atomic_bool stop{false};
    std::atomic_int errorCode{0}, retries{0}, aps{0}, strMode{0};
    std::atomic<qint64> position{0};
    std::promise<quint16> portPromise;
    auto portFuture = portPromise.get_future();
    std::thread server([&]() {
        QTcpServer listener;
        if (!listener.listen(QHostAddress::LocalHost, 0))
        {
            portPromise.set_value(0);
            return;
        }
        portPromise.set_value(listener.serverPort());
        while (!stop && !listener.hasPendingConnections())
            listener.waitForNewConnection(100);
        if (stop)
            return;
        auto socket = listener.nextPendingConnection();
        QByteArray buffer;
        while (!stop && socket->state() == QAbstractSocket::ConnectedState)
        {
            if (!socket->bytesAvailable())
                socket->waitForReadyRead(100);
            buffer += socket->readAll();
            while (buffer.contains('\n'))
            {
                auto end = buffer.indexOf('\n');
                QString cmd = QString::fromLatin1(buffer.left(end)).trimmed();
                buffer.remove(0, end + 1);
                QString token = cmd.section('/', 0, 0), response = "C " + token;
                if (cmd.startsWith("RDP"))
                    response += " " + QString::number(position.load());
                else if (cmd.startsWith("STR"))
                    response += strMode == 2 ? " unknown" : QString(" %1 0 0 0 0 0").arg(strMode.load());
                else if (cmd.startsWith("WTB") && retries > 0)
                {
                    --retries;
                    response = "E " + token + " " + QString::number(errorCode.load());
                }
                else if (cmd.startsWith("APS"))
                {
                    ++aps;
                    position = cmd.section('/', 2, 2).toLongLong();
                }
                else if (cmd.startsWith("STP"))
                    strMode = 0;
                socket->write(response.toLatin1() + "\r\n");
                socket->waitForBytesWritten(1000);
            }
        }
        delete socket;
    });
    int result = 0;
    try
    {
        auto insist = [](bool b, const char *s) {
            if (!b)
                throw std::runtime_error(s);
        };
        const auto port = portFuture.get();
        insist(port != 0, "Loopback simulator failed");
        ct::KohzuController motor;
        motor.addAxis(15);
        motor.setIdlePollingEnabled(false);
        bool connected = false, ok = false;
        QString wanted;
        QObject::connect(&motor, &ct::KohzuController::connectionChanged, [&](bool b) { connected = b; });
        QObject::connect(&motor, &ct::KohzuController::commandFinished,
                         [&](const QString &label, bool success, const QString &) {
                             if (label == wanted)
                                 ok = success;
                         });
        motor.connectToController("127.0.0.1", port);
        insist(connected, "Simulator connection failed");
        auto move = [&](int speed, qint64 target) {
            wanted = "move";
            ok = false;
            motor.configureSpeedTableAndMoveAbsolute(15, target, 8, 10, speed, 10, 10, 2, "speed", "move");
        };
        errorCode = 605;
        retries = 1;
        move(100, 1234);
        insist(ok && aps == 1, "Transient 605 recovery failed");
        wanted = "verify";
        ok = false;
        motor.verifyAxisPosition(15, 1234, 1, wanted);
        insist(ok, "Idle + actual RDP verification failed");
        motor.verifyAxisPosition(15, 9999, 1, wanted);
        insist(!ok, "Wrong position accepted");
        strMode = 2;
        motor.verifyAxisPosition(15, 1234, 1, wanted);
        insist(!ok, "Unknown STR accepted");
        strMode = 0;
        errorCode = 803;
        retries = 1;
        move(101, 2345);
        insist(ok && aps == 2, "Transient 803 recovery failed");
        errorCode = 605;
        retries = 10;
        move(102, 3456);
        insist(!ok && aps == 2, "APS sent after WTB failure");
        retries = 0;
        wanted = "Kohzu stop all";
        ok = false;
        motor.stopAllAxes();
        insist(ok, "Stop failed");
        move(102, 3456);
        insist(ok && aps == 3, "Motion did not recover after Stop");
        motor.shutdown();
        std::cout << "PASS: actual RDP, idle/unknown STR, 605/803, WTB failure blocks APS, controlled Stop recovery\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        result = 1;
    }
    stop = true;
    server.join();
    return result;
}
