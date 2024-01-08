/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.5.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <mouseevent.h>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout_5;
    QStackedWidget *stackedWidget;
    QWidget *Home;
    QHBoxLayout *horizontalLayout;
    QHBoxLayout *horizontalLayout_6;
    QSpacerItem *horizontalSpacer_5;
    QVBoxLayout *verticalLayout;
    QSpacerItem *verticalSpacer_4;
    QPushButton *upload;
    QSpacerItem *verticalSpacer_6;
    QPushButton *about;
    QSpacerItem *verticalSpacer_8;
    QPushButton *discription;
    QSpacerItem *verticalSpacer_5;
    QSpacerItem *horizontalSpacer_6;
    QWidget *Discription;
    QVBoxLayout *verticalLayout_6;
    QGridLayout *gridLayout;
    QSpacerItem *horizontalSpacer_9;
    QSpacerItem *horizontalSpacer_10;
    QSpacerItem *horizontalSpacer_7;
    QPushButton *return1;
    QSpacerItem *horizontalSpacer_11;
    QTextBrowser *textBrowser_2;
    QWidget *About;
    QGridLayout *gridLayout_3;
    QGridLayout *gridLayout_2;
    QSpacerItem *horizontalSpacer_8;
    QPushButton *return2;
    QSpacerItem *horizontalSpacer_13;
    QSpacerItem *horizontalSpacer_12;
    QTextBrowser *textBrowser;
    QWidget *showImageRGB;
    QHBoxLayout *horizontalLayout_8;
    QHBoxLayout *horizontalLayout_14;
    QHBoxLayout *horizontalLayout_12;
    QLabel *originImage;
    MouseEvent *starIImage;
    QFrame *line_2;
    QGridLayout *gridLayout_4;
    QSlider *R;
    QSpacerItem *verticalSpacer_3;
    QSlider *B;
    QPushButton *ok;
    QLabel *Glabel_2;
    QLabel *Rlabel;
    QLabel *lblMouse_Current_Pos;
    QLabel *Blabel;
    QSpacerItem *verticalSpacer;
    QLabel *Glabel;
    QLabel *lblMouse_Current_Event;
    QLabel *Blabel_2;
    QLabel *Rlabel_2;
    QSlider *G;
    QWidget *ShowStarTrail;
    QGridLayout *gridLayout_6;
    QHBoxLayout *horizontalLayout_3;
    QTabWidget *tabWidget_origin;
    QWidget *originImage_2;
    QHBoxLayout *horizontalLayout_2;
    QLabel *E_origin;
    QWidget *FrontandBack;
    QHBoxLayout *horizontalLayout_4;
    QLabel *E_frontandback;
    QTabWidget *tabWidget_star;
    QWidget *star;
    QHBoxLayout *horizontalLayout_5;
    QLabel *E_star;
    QWidget *starTrail;
    QHBoxLayout *horizontalLayout_9;
    QLabel *E_starTrail;
    QTabWidget *tabWidget_result;
    QWidget *pngResult;
    QHBoxLayout *horizontalLayout_10;
    QLabel *E_png;
    QHBoxLayout *horizontalLayout_13;
    QSpacerItem *horizontalSpacer_15;
    QLabel *label;
    QLabel *passTime;
    QLabel *label_4;
    QSpacerItem *horizontalSpacer_16;
    QPushButton *pngSaveBtn;
    QPushButton *gifSaveBtn;
    QHBoxLayout *horizontalLayout_7;
    QSpacerItem *horizontalSpacer_2;
    QLabel *TrailLength;
    QSpacerItem *horizontalSpacer_14;
    QSlider *TrailLengthSlider;
    QSpacerItem *horizontalSpacer_3;
    QLabel *label_2;
    QComboBox *TrailShape;
    QSpacerItem *horizontalSpacer;
    QPushButton *startGenerate;
    QPushButton *back2Home;
    QSpacerItem *horizontalSpacer_4;
    QFrame *line;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(1254, 769);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        gridLayout_5 = new QGridLayout(centralwidget);
        gridLayout_5->setObjectName("gridLayout_5");
        stackedWidget = new QStackedWidget(centralwidget);
        stackedWidget->setObjectName("stackedWidget");
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(stackedWidget->sizePolicy().hasHeightForWidth());
        stackedWidget->setSizePolicy(sizePolicy1);
        Home = new QWidget();
        Home->setObjectName("Home");
        Home->setMinimumSize(QSize(100, 50));
        horizontalLayout = new QHBoxLayout(Home);
        horizontalLayout->setObjectName("horizontalLayout");
        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName("horizontalLayout_6");
        horizontalLayout_6->setContentsMargins(12, -1, 12, 12);
        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Preferred, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_5);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName("verticalLayout");
        verticalLayout->setContentsMargins(20, 20, 20, 20);
        verticalSpacer_4 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer_4);

        upload = new QPushButton(Home);
        upload->setObjectName("upload");
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(upload->sizePolicy().hasHeightForWidth());
        upload->setSizePolicy(sizePolicy2);
        upload->setMaximumSize(QSize(300, 60));
        QFont font;
        font.setFamilies({QString::fromUtf8("\346\272\220\346\263\211\345\234\223\351\253\224 M")});
        font.setPointSize(36);
        upload->setFont(font);

        verticalLayout->addWidget(upload);

        verticalSpacer_6 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer_6);

        about = new QPushButton(Home);
        about->setObjectName("about");
        sizePolicy2.setHeightForWidth(about->sizePolicy().hasHeightForWidth());
        about->setSizePolicy(sizePolicy2);
        about->setMaximumSize(QSize(300, 60));
        about->setFont(font);

        verticalLayout->addWidget(about);

        verticalSpacer_8 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer_8);

        discription = new QPushButton(Home);
        discription->setObjectName("discription");
        sizePolicy2.setHeightForWidth(discription->sizePolicy().hasHeightForWidth());
        discription->setSizePolicy(sizePolicy2);
        discription->setMaximumSize(QSize(300, 60));
        discription->setFont(font);

        verticalLayout->addWidget(discription);

        verticalSpacer_5 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer_5);


        horizontalLayout_6->addLayout(verticalLayout);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Preferred, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_6);


        horizontalLayout->addLayout(horizontalLayout_6);

        stackedWidget->addWidget(Home);
        Discription = new QWidget();
        Discription->setObjectName("Discription");
        verticalLayout_6 = new QVBoxLayout(Discription);
        verticalLayout_6->setObjectName("verticalLayout_6");
        gridLayout = new QGridLayout();
        gridLayout->setObjectName("gridLayout");
        horizontalSpacer_9 = new QSpacerItem(60, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_9, 1, 0, 1, 1);

        horizontalSpacer_10 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_10, 0, 3, 2, 1);

        horizontalSpacer_7 = new QSpacerItem(800, 20, QSizePolicy::Preferred, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_7, 2, 1, 3, 2);

        return1 = new QPushButton(Discription);
        return1->setObjectName("return1");
        QFont font1;
        font1.setFamilies({QString::fromUtf8("\346\272\220\346\263\211\345\234\223\351\253\224 R")});
        font1.setPointSize(24);
        return1->setFont(font1);

        gridLayout->addWidget(return1, 2, 3, 3, 1);

        horizontalSpacer_11 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_11, 2, 0, 3, 1);

        textBrowser_2 = new QTextBrowser(Discription);
        textBrowser_2->setObjectName("textBrowser_2");

        gridLayout->addWidget(textBrowser_2, 1, 1, 1, 2);


        verticalLayout_6->addLayout(gridLayout);

        stackedWidget->addWidget(Discription);
        About = new QWidget();
        About->setObjectName("About");
        gridLayout_3 = new QGridLayout(About);
        gridLayout_3->setObjectName("gridLayout_3");
        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName("gridLayout_2");
        horizontalSpacer_8 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_8, 1, 0, 1, 5);

        return2 = new QPushButton(About);
        return2->setObjectName("return2");
        return2->setFont(font1);

        gridLayout_2->addWidget(return2, 1, 5, 1, 1);

        horizontalSpacer_13 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_13, 0, 5, 1, 1);

        horizontalSpacer_12 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_12, 0, 0, 1, 1);

        textBrowser = new QTextBrowser(About);
        textBrowser->setObjectName("textBrowser");
        textBrowser->setMinimumSize(QSize(600, 300));

        gridLayout_2->addWidget(textBrowser, 0, 1, 1, 4);


        gridLayout_3->addLayout(gridLayout_2, 0, 0, 1, 1);

        stackedWidget->addWidget(About);
        showImageRGB = new QWidget();
        showImageRGB->setObjectName("showImageRGB");
        horizontalLayout_8 = new QHBoxLayout(showImageRGB);
        horizontalLayout_8->setObjectName("horizontalLayout_8");
        horizontalLayout_14 = new QHBoxLayout();
        horizontalLayout_14->setObjectName("horizontalLayout_14");
        horizontalLayout_14->setContentsMargins(12, -1, 12, -1);
        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName("horizontalLayout_12");
        horizontalLayout_12->setSizeConstraint(QLayout::SetDefaultConstraint);
        horizontalLayout_12->setContentsMargins(6, 6, 6, 6);
        originImage = new QLabel(showImageRGB);
        originImage->setObjectName("originImage");
        QSizePolicy sizePolicy3(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(originImage->sizePolicy().hasHeightForWidth());
        originImage->setSizePolicy(sizePolicy3);
        originImage->setMinimumSize(QSize(400, 0));
        originImage->setMaximumSize(QSize(750, 16777215));
        originImage->setMouseTracking(false);
        originImage->setStyleSheet(QString::fromUtf8("background-color: rgb(255, 255, 255);"));
        originImage->setFrameShape(QFrame::Box);
        originImage->setScaledContents(true);
        originImage->setAlignment(Qt::AlignCenter);

        horizontalLayout_12->addWidget(originImage);

        starIImage = new MouseEvent(showImageRGB);
        starIImage->setObjectName("starIImage");
        sizePolicy3.setHeightForWidth(starIImage->sizePolicy().hasHeightForWidth());
        starIImage->setSizePolicy(sizePolicy3);
        starIImage->setMinimumSize(QSize(400, 0));
        starIImage->setMaximumSize(QSize(16777215, 16777215));
        starIImage->setMouseTracking(true);
        starIImage->setAutoFillBackground(false);
        starIImage->setStyleSheet(QString::fromUtf8("background-color: rgb(255, 255, 255);"));
        starIImage->setFrameShape(QFrame::Box);
        starIImage->setScaledContents(true);
        starIImage->setAlignment(Qt::AlignCenter);

        horizontalLayout_12->addWidget(starIImage);

        horizontalLayout_12->setStretch(1, 1);

        horizontalLayout_14->addLayout(horizontalLayout_12);

        line_2 = new QFrame(showImageRGB);
        line_2->setObjectName("line_2");
        line_2->setFrameShape(QFrame::VLine);
        line_2->setFrameShadow(QFrame::Sunken);

        horizontalLayout_14->addWidget(line_2);

        gridLayout_4 = new QGridLayout();
        gridLayout_4->setObjectName("gridLayout_4");
        gridLayout_4->setSizeConstraint(QLayout::SetMaximumSize);
        gridLayout_4->setContentsMargins(15, 15, 15, 15);
        R = new QSlider(showImageRGB);
        R->setObjectName("R");
        R->setMinimumSize(QSize(200, 0));
        R->setMaximumSize(QSize(200, 16777215));
        R->setMaximum(255);
        R->setValue(60);
        R->setTracking(false);
        R->setOrientation(Qt::Horizontal);
        R->setTickPosition(QSlider::TicksBelow);

        gridLayout_4->addWidget(R, 3, 3, 1, 1);

        verticalSpacer_3 = new QSpacerItem(13, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_4->addItem(verticalSpacer_3, 4, 5, 2, 2);

        B = new QSlider(showImageRGB);
        B->setObjectName("B");
        B->setMinimumSize(QSize(20, 0));
        B->setMaximumSize(QSize(200, 16777215));
        B->setAutoFillBackground(false);
        B->setMaximum(255);
        B->setValue(174);
        B->setTracking(false);
        B->setOrientation(Qt::Horizontal);
        B->setTickPosition(QSlider::TicksBelow);

        gridLayout_4->addWidget(B, 0, 3, 1, 1);

        ok = new QPushButton(showImageRGB);
        ok->setObjectName("ok");
        QFont font2;
        font2.setFamilies({QString::fromUtf8("\346\272\220\346\263\211\345\234\223\351\253\224 M")});
        font2.setPointSize(20);
        ok->setFont(font2);

        gridLayout_4->addWidget(ok, 6, 5, 1, 2);

        Glabel_2 = new QLabel(showImageRGB);
        Glabel_2->setObjectName("Glabel_2");
        QSizePolicy sizePolicy4(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(Glabel_2->sizePolicy().hasHeightForWidth());
        Glabel_2->setSizePolicy(sizePolicy4);
        Glabel_2->setMinimumSize(QSize(50, 0));
        Glabel_2->setMaximumSize(QSize(50, 16777215));
        Glabel_2->setFont(font2);

        gridLayout_4->addWidget(Glabel_2, 1, 5, 1, 2);

        Rlabel = new QLabel(showImageRGB);
        Rlabel->setObjectName("Rlabel");
        sizePolicy4.setHeightForWidth(Rlabel->sizePolicy().hasHeightForWidth());
        Rlabel->setSizePolicy(sizePolicy4);
        Rlabel->setFont(font2);

        gridLayout_4->addWidget(Rlabel, 3, 0, 1, 3);

        lblMouse_Current_Pos = new QLabel(showImageRGB);
        lblMouse_Current_Pos->setObjectName("lblMouse_Current_Pos");

        gridLayout_4->addWidget(lblMouse_Current_Pos, 4, 3, 1, 1);

        Blabel = new QLabel(showImageRGB);
        Blabel->setObjectName("Blabel");
        Blabel->setFont(font2);

        gridLayout_4->addWidget(Blabel, 0, 0, 1, 3);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_4->addItem(verticalSpacer, 4, 0, 2, 3);

        Glabel = new QLabel(showImageRGB);
        Glabel->setObjectName("Glabel");
        sizePolicy4.setHeightForWidth(Glabel->sizePolicy().hasHeightForWidth());
        Glabel->setSizePolicy(sizePolicy4);
        Glabel->setFont(font2);

        gridLayout_4->addWidget(Glabel, 1, 0, 1, 3);

        lblMouse_Current_Event = new QLabel(showImageRGB);
        lblMouse_Current_Event->setObjectName("lblMouse_Current_Event");

        gridLayout_4->addWidget(lblMouse_Current_Event, 5, 3, 1, 1);

        Blabel_2 = new QLabel(showImageRGB);
        Blabel_2->setObjectName("Blabel_2");
        sizePolicy4.setHeightForWidth(Blabel_2->sizePolicy().hasHeightForWidth());
        Blabel_2->setSizePolicy(sizePolicy4);
        Blabel_2->setMinimumSize(QSize(50, 0));
        Blabel_2->setMaximumSize(QSize(50, 16777215));
        Blabel_2->setFont(font2);

        gridLayout_4->addWidget(Blabel_2, 0, 5, 1, 1);

        Rlabel_2 = new QLabel(showImageRGB);
        Rlabel_2->setObjectName("Rlabel_2");
        sizePolicy4.setHeightForWidth(Rlabel_2->sizePolicy().hasHeightForWidth());
        Rlabel_2->setSizePolicy(sizePolicy4);
        Rlabel_2->setMinimumSize(QSize(50, 0));
        Rlabel_2->setMaximumSize(QSize(50, 16777215));
        Rlabel_2->setFont(font2);

        gridLayout_4->addWidget(Rlabel_2, 3, 5, 1, 1);

        G = new QSlider(showImageRGB);
        G->setObjectName("G");
        G->setMinimumSize(QSize(20, 0));
        G->setMaximumSize(QSize(200, 16777215));
        G->setMaximum(255);
        G->setValue(154);
        G->setTracking(false);
        G->setOrientation(Qt::Horizontal);
        G->setTickPosition(QSlider::TicksBelow);

        gridLayout_4->addWidget(G, 1, 3, 1, 1);


        horizontalLayout_14->addLayout(gridLayout_4);

        horizontalLayout_14->setStretch(0, 5);
        horizontalLayout_14->setStretch(2, 1);

        horizontalLayout_8->addLayout(horizontalLayout_14);

        stackedWidget->addWidget(showImageRGB);
        ShowStarTrail = new QWidget();
        ShowStarTrail->setObjectName("ShowStarTrail");
        gridLayout_6 = new QGridLayout(ShowStarTrail);
        gridLayout_6->setObjectName("gridLayout_6");
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName("horizontalLayout_3");
        horizontalLayout_3->setContentsMargins(3, 3, 3, 3);
        tabWidget_origin = new QTabWidget(ShowStarTrail);
        tabWidget_origin->setObjectName("tabWidget_origin");
        sizePolicy1.setHeightForWidth(tabWidget_origin->sizePolicy().hasHeightForWidth());
        tabWidget_origin->setSizePolicy(sizePolicy1);
        tabWidget_origin->setMinimumSize(QSize(400, 600));
        tabWidget_origin->setStyleSheet(QString::fromUtf8("QTabBar::tab(width: 120)"));
        originImage_2 = new QWidget();
        originImage_2->setObjectName("originImage_2");
        horizontalLayout_2 = new QHBoxLayout(originImage_2);
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        E_origin = new QLabel(originImage_2);
        E_origin->setObjectName("E_origin");
        sizePolicy3.setHeightForWidth(E_origin->sizePolicy().hasHeightForWidth());
        E_origin->setSizePolicy(sizePolicy3);
        E_origin->setMinimumSize(QSize(300, 0));
        E_origin->setMaximumSize(QSize(650, 16777215));
        E_origin->setStyleSheet(QString::fromUtf8("background-color: rgb(255, 255, 255);"));
        E_origin->setScaledContents(true);

        horizontalLayout_2->addWidget(E_origin);

        tabWidget_origin->addTab(originImage_2, QString());
        FrontandBack = new QWidget();
        FrontandBack->setObjectName("FrontandBack");
        horizontalLayout_4 = new QHBoxLayout(FrontandBack);
        horizontalLayout_4->setObjectName("horizontalLayout_4");
        E_frontandback = new QLabel(FrontandBack);
        E_frontandback->setObjectName("E_frontandback");
        sizePolicy3.setHeightForWidth(E_frontandback->sizePolicy().hasHeightForWidth());
        E_frontandback->setSizePolicy(sizePolicy3);
        E_frontandback->setMinimumSize(QSize(300, 0));
        E_frontandback->setMaximumSize(QSize(650, 16777215));
        E_frontandback->setStyleSheet(QString::fromUtf8("background-color: rgb(255, 255, 255);"));
        E_frontandback->setScaledContents(true);

        horizontalLayout_4->addWidget(E_frontandback);

        tabWidget_origin->addTab(FrontandBack, QString());

        horizontalLayout_3->addWidget(tabWidget_origin);

        tabWidget_star = new QTabWidget(ShowStarTrail);
        tabWidget_star->setObjectName("tabWidget_star");
        sizePolicy1.setHeightForWidth(tabWidget_star->sizePolicy().hasHeightForWidth());
        tabWidget_star->setSizePolicy(sizePolicy1);
        tabWidget_star->setMinimumSize(QSize(400, 600));
        star = new QWidget();
        star->setObjectName("star");
        horizontalLayout_5 = new QHBoxLayout(star);
        horizontalLayout_5->setObjectName("horizontalLayout_5");
        E_star = new QLabel(star);
        E_star->setObjectName("E_star");
        sizePolicy3.setHeightForWidth(E_star->sizePolicy().hasHeightForWidth());
        E_star->setSizePolicy(sizePolicy3);
        E_star->setMinimumSize(QSize(300, 0));
        E_star->setMaximumSize(QSize(650, 16777215));
        E_star->setStyleSheet(QString::fromUtf8("background-color: rgb(255, 255, 255);"));
        E_star->setScaledContents(true);

        horizontalLayout_5->addWidget(E_star);

        tabWidget_star->addTab(star, QString());
        starTrail = new QWidget();
        starTrail->setObjectName("starTrail");
        horizontalLayout_9 = new QHBoxLayout(starTrail);
        horizontalLayout_9->setObjectName("horizontalLayout_9");
        E_starTrail = new QLabel(starTrail);
        E_starTrail->setObjectName("E_starTrail");
        QSizePolicy sizePolicy5(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(E_starTrail->sizePolicy().hasHeightForWidth());
        E_starTrail->setSizePolicy(sizePolicy5);
        E_starTrail->setMinimumSize(QSize(400, 0));
        E_starTrail->setMaximumSize(QSize(16777215, 16777215));
        E_starTrail->setStyleSheet(QString::fromUtf8("background-color: rgb(255, 255, 255);"));
        E_starTrail->setScaledContents(true);
        E_starTrail->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        horizontalLayout_9->addWidget(E_starTrail);

        tabWidget_star->addTab(starTrail, QString());

        horizontalLayout_3->addWidget(tabWidget_star);

        tabWidget_result = new QTabWidget(ShowStarTrail);
        tabWidget_result->setObjectName("tabWidget_result");
        sizePolicy1.setHeightForWidth(tabWidget_result->sizePolicy().hasHeightForWidth());
        tabWidget_result->setSizePolicy(sizePolicy1);
        tabWidget_result->setMinimumSize(QSize(400, 600));
        pngResult = new QWidget();
        pngResult->setObjectName("pngResult");
        horizontalLayout_10 = new QHBoxLayout(pngResult);
        horizontalLayout_10->setObjectName("horizontalLayout_10");
        E_png = new QLabel(pngResult);
        E_png->setObjectName("E_png");
        sizePolicy3.setHeightForWidth(E_png->sizePolicy().hasHeightForWidth());
        E_png->setSizePolicy(sizePolicy3);
        E_png->setMinimumSize(QSize(400, 0));
        E_png->setMaximumSize(QSize(650, 16777215));
        E_png->setStyleSheet(QString::fromUtf8("background-color: rgb(255, 255, 255);"));
        E_png->setScaledContents(true);

        horizontalLayout_10->addWidget(E_png);

        tabWidget_result->addTab(pngResult, QString());

        horizontalLayout_3->addWidget(tabWidget_result);

        horizontalLayout_3->setStretch(0, 1);
        horizontalLayout_3->setStretch(1, 1);
        horizontalLayout_3->setStretch(2, 1);

        gridLayout_6->addLayout(horizontalLayout_3, 3, 0, 1, 1);

        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setObjectName("horizontalLayout_13");
        horizontalLayout_13->setContentsMargins(-1, 6, -1, 3);
        horizontalSpacer_15 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_13->addItem(horizontalSpacer_15);

        label = new QLabel(ShowStarTrail);
        label->setObjectName("label");
        QFont font3;
        font3.setFamilies({QString::fromUtf8("\346\272\220\346\263\211\345\234\223\351\253\224 M")});
        font3.setPointSize(22);
        label->setFont(font3);
        label->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_13->addWidget(label);

        passTime = new QLabel(ShowStarTrail);
        passTime->setObjectName("passTime");
        passTime->setMinimumSize(QSize(198, 0));
        passTime->setMaximumSize(QSize(200, 16777215));
        passTime->setFont(font3);
        passTime->setAlignment(Qt::AlignCenter);

        horizontalLayout_13->addWidget(passTime);

        label_4 = new QLabel(ShowStarTrail);
        label_4->setObjectName("label_4");
        label_4->setFont(font3);

        horizontalLayout_13->addWidget(label_4);

        horizontalSpacer_16 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_13->addItem(horizontalSpacer_16);

        pngSaveBtn = new QPushButton(ShowStarTrail);
        pngSaveBtn->setObjectName("pngSaveBtn");
        QFont font4;
        font4.setFamilies({QString::fromUtf8("\346\272\220\346\263\211\345\234\223\351\253\224 R")});
        font4.setPointSize(14);
        pngSaveBtn->setFont(font4);

        horizontalLayout_13->addWidget(pngSaveBtn);

        gifSaveBtn = new QPushButton(ShowStarTrail);
        gifSaveBtn->setObjectName("gifSaveBtn");
        gifSaveBtn->setFont(font4);

        horizontalLayout_13->addWidget(gifSaveBtn);


        gridLayout_6->addLayout(horizontalLayout_13, 5, 0, 1, 1);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(12);
        horizontalLayout_7->setObjectName("horizontalLayout_7");
        horizontalLayout_7->setContentsMargins(3, 3, 3, 3);
        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_2);

        TrailLength = new QLabel(ShowStarTrail);
        TrailLength->setObjectName("TrailLength");
        TrailLength->setMinimumSize(QSize(250, 0));
        TrailLength->setMaximumSize(QSize(250, 16777215));
        QFont font5;
        font5.setFamilies({QString::fromUtf8("\346\272\220\346\263\211\345\234\223\351\253\224 M")});
        font5.setPointSize(16);
        TrailLength->setFont(font5);
        TrailLength->setAlignment(Qt::AlignCenter);

        horizontalLayout_7->addWidget(TrailLength);

        horizontalSpacer_14 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_14);

        TrailLengthSlider = new QSlider(ShowStarTrail);
        TrailLengthSlider->setObjectName("TrailLengthSlider");
        sizePolicy3.setHeightForWidth(TrailLengthSlider->sizePolicy().hasHeightForWidth());
        TrailLengthSlider->setSizePolicy(sizePolicy3);
        TrailLengthSlider->setMinimumSize(QSize(250, 0));
        TrailLengthSlider->setMaximumSize(QSize(250, 16777215));
        TrailLengthSlider->setMaximum(1440);
        TrailLengthSlider->setSingleStep(30);
        TrailLengthSlider->setPageStep(30);
        TrailLengthSlider->setOrientation(Qt::Horizontal);
        TrailLengthSlider->setTickPosition(QSlider::TicksAbove);

        horizontalLayout_7->addWidget(TrailLengthSlider);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_3);

        label_2 = new QLabel(ShowStarTrail);
        label_2->setObjectName("label_2");
        label_2->setFont(font5);

        horizontalLayout_7->addWidget(label_2);

        TrailShape = new QComboBox(ShowStarTrail);
        TrailShape->addItem(QString());
        TrailShape->addItem(QString());
        TrailShape->addItem(QString());
        TrailShape->addItem(QString());
        TrailShape->setObjectName("TrailShape");
        QFont font6;
        font6.setFamilies({QString::fromUtf8("\346\272\220\346\263\211\345\234\223\351\253\224 R")});
        font6.setPointSize(16);
        TrailShape->setFont(font6);

        horizontalLayout_7->addWidget(TrailShape);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer);

        startGenerate = new QPushButton(ShowStarTrail);
        startGenerate->setObjectName("startGenerate");
        startGenerate->setFont(font2);

        horizontalLayout_7->addWidget(startGenerate);

        back2Home = new QPushButton(ShowStarTrail);
        back2Home->setObjectName("back2Home");
        back2Home->setFont(font2);

        horizontalLayout_7->addWidget(back2Home);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_4);

        horizontalLayout_7->setStretch(5, 2);
        horizontalLayout_7->setStretch(6, 4);
        horizontalLayout_7->setStretch(8, 2);
        horizontalLayout_7->setStretch(9, 2);

        gridLayout_6->addLayout(horizontalLayout_7, 1, 0, 1, 1);

        line = new QFrame(ShowStarTrail);
        line->setObjectName("line");
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        gridLayout_6->addWidget(line, 2, 0, 1, 1);

        stackedWidget->addWidget(ShowStarTrail);

        gridLayout_5->addWidget(stackedWidget, 0, 0, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);
        QObject::connect(upload, SIGNAL(clicked()), MainWindow, SLOT(on_uploadButton_clicked()));

        stackedWidget->setCurrentIndex(0);
        tabWidget_origin->setCurrentIndex(1);
        tabWidget_star->setCurrentIndex(0);
        tabWidget_result->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        upload->setText(QCoreApplication::translate("MainWindow", "\350\243\275\344\275\234\346\230\237\350\273\214", nullptr));
        about->setText(QCoreApplication::translate("MainWindow", "\351\227\234\346\226\274\344\275\234\350\200\205", nullptr));
        discription->setText(QCoreApplication::translate("MainWindow", "\344\275\277\347\224\250\350\252\252\346\230\216", nullptr));
        return1->setText(QCoreApplication::translate("MainWindow", "\345\233\236\345\210\260\351\246\226\351\240\201", nullptr));
        textBrowser_2->setHtml(QCoreApplication::translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Microsoft JhengHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:24pt;\">\350\247\243\350\252\252\351\200\231\345\200\213App\346\200\216\351\272\274\344\275\277\347\224\250</span></p></body></html>", nullptr));
        return2->setText(QCoreApplication::translate("MainWindow", "\345\233\236\345\210\260\351\246\226\351\240\201", nullptr));
        textBrowser->setHtml(QCoreApplication::translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Microsoft JhengHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:24pt;\">Made by \346\261\237\344\275\251\350\223\211 \350\230\207\346\237\217\345\207\261</span></p></body></html>", nullptr));
        originImage->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
        starIImage->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
        ok->setText(QCoreApplication::translate("MainWindow", "OK", nullptr));
        Glabel_2->setText(QCoreApplication::translate("MainWindow", "154", nullptr));
        Rlabel->setText(QCoreApplication::translate("MainWindow", "R", nullptr));
        lblMouse_Current_Pos->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
        Blabel->setText(QCoreApplication::translate("MainWindow", "B", nullptr));
        Glabel->setText(QCoreApplication::translate("MainWindow", "G", nullptr));
        lblMouse_Current_Event->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
        Blabel_2->setText(QCoreApplication::translate("MainWindow", "174", nullptr));
        Rlabel_2->setText(QCoreApplication::translate("MainWindow", "60", nullptr));
        E_origin->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
        tabWidget_origin->setTabText(tabWidget_origin->indexOf(originImage_2), QCoreApplication::translate("MainWindow", "\345\216\237\345\247\213\345\234\226", nullptr));
        E_frontandback->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
        tabWidget_origin->setTabText(tabWidget_origin->indexOf(FrontandBack), QCoreApplication::translate("MainWindow", "\345\210\206\351\232\224\345\211\215\345\276\214\346\231\257", nullptr));
        E_star->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
        tabWidget_star->setTabText(tabWidget_star->indexOf(star), QCoreApplication::translate("MainWindow", "\346\230\237\351\273\236", nullptr));
        E_starTrail->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
        tabWidget_star->setTabText(tabWidget_star->indexOf(starTrail), QCoreApplication::translate("MainWindow", "\346\230\237\350\273\214", nullptr));
        E_png->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
        tabWidget_result->setTabText(tabWidget_result->indexOf(pngResult), QCoreApplication::translate("MainWindow", "png\345\234\226", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "\346\211\200\350\262\273\346\231\202\351\226\223\357\274\232", nullptr));
        passTime->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "\347\247\222", nullptr));
        pngSaveBtn->setText(QCoreApplication::translate("MainWindow", "\345\204\262\345\255\230png", nullptr));
        gifSaveBtn->setText(QCoreApplication::translate("MainWindow", "\345\204\262\345\255\230gif", nullptr));
        TrailLength->setText(QCoreApplication::translate("MainWindow", "\346\233\235\345\205\211\346\231\202\351\225\267\357\274\2320 \345\260\217\346\231\2020 \345\210\206\351\220\230", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "\346\230\237\350\273\214\345\275\242\347\213\200", nullptr));
        TrailShape->setItemText(0, QCoreApplication::translate("MainWindow", "\345\234\223\345\275\242", nullptr));
        TrailShape->setItemText(1, QCoreApplication::translate("MainWindow", "\350\236\272\346\227\213\347\213\200", nullptr));
        TrailShape->setItemText(2, QCoreApplication::translate("MainWindow", "\346\224\276\345\260\204\347\213\200", nullptr));
        TrailShape->setItemText(3, QCoreApplication::translate("MainWindow", "\345\275\227\346\230\237\346\222\236\345\234\260\347\220\203", nullptr));

        startGenerate->setText(QCoreApplication::translate("MainWindow", "\351\226\213\345\247\213\347\224\237\346\210\220", nullptr));
        back2Home->setText(QCoreApplication::translate("MainWindow", "\347\271\274\347\272\214\347\224\237\346\210\220", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
