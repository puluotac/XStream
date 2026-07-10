#include "ui/MainWindow.h"

#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QMetaType>
#include <QTextStream>

#include <opencv2/core.hpp>

#include <memory>

Q_DECLARE_METATYPE(cv::Mat)

namespace
{

void writeStartupTrace(const QString& text)
{
    QFile file(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("ct_startup_trace.log")));
    if (file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
    {
        QTextStream stream(&file);
        stream << QDateTime::currentDateTime().toString(Qt::ISODateWithMs) << " " << text << '\n';
    }
}

} // namespace

int main(int argc, char* argv[])
{
    writeStartupTrace(QStringLiteral("main enter"));
    QApplication app(argc, argv);
    writeStartupTrace(QStringLiteral("QApplication constructed"));
    qRegisterMetaType<cv::Mat>("cv::Mat");
    writeStartupTrace(QStringLiteral("metatypes registered; constructing MainWindow"));
    auto window = std::make_unique<ct::MainWindow>();
    writeStartupTrace(QStringLiteral("MainWindow constructed; showing"));
    window->show();
    const int rc = app.exec();
    writeStartupTrace(QStringLiteral("app exec returned %1").arg(rc));
    window.reset();
    writeStartupTrace(QStringLiteral("MainWindow destroyed"));
    return rc;
}
