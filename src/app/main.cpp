#include "ui/MainWindow.h"

#include <QApplication>
#include <QMetaType>

#include <opencv2/core.hpp>

Q_DECLARE_METATYPE(cv::Mat)

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    qRegisterMetaType<cv::Mat>("cv::Mat");
    ct::MainWindow window;
    window.show();
    return app.exec();
}
