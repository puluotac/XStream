#include "ui/MainWindow.h"

#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QMetaType>
#include <QTextStream>
#include <QTimer>
#include <QTabWidget>
#include <QPushButton>

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
    QApplication app(argc, argv);
    writeStartupTrace(QStringLiteral("main enter; QApplication constructed"));
    qRegisterMetaType<cv::Mat>("cv::Mat");
    writeStartupTrace(QStringLiteral("metatypes registered; constructing MainWindow"));
    auto window = std::make_unique<ct::MainWindow>();
    writeStartupTrace(QStringLiteral("MainWindow constructed; showing"));
    window->show();
    if (qEnvironmentVariableIsSet("CT_UI_SMOKE"))
    {
        QTimer::singleShot(300, window.get(), [&]() {
            auto* page = window->findChild<QWidget*>("mosaicPage");
            if (!page) { app.exit(3); return; }
            auto* tabs = page->parentWidget()->parentWidget();
            if (auto* tab = qobject_cast<QTabWidget*>(tabs)) tab->setCurrentWidget(page);
            window->grab().save(qEnvironmentVariable("CT_UI_SMOKE") + "/mosaic_ui.png");
            if (qEnvironmentVariableIsSet("CT_MOSAIC_QUEUE_TEST")) window->runMosaicQueueSmoke();
            else QTimer::singleShot(300, window.get(), &QWidget::close);
        });
    }
    const int rc = app.exec();
    writeStartupTrace(QStringLiteral("app exec returned %1").arg(rc));
    window.reset();
    writeStartupTrace(QStringLiteral("MainWindow destroyed"));
    return rc;
}
