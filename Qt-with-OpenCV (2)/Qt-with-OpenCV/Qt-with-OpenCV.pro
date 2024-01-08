QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    mouseevent.cpp

HEADERS += \
    mainwindow.h \
    mouseevent.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(debug, debug | release):{
LIBS +=  -LC:/opencv24/build/x64/vc14/lib \
-lopencv_core2413d \
-lopencv_imgproc2413d \
-lopencv_highgui2413d \
-lopencv_ml2413d \
-lopencv_video2413d \
-lopencv_features2d2413d \
-lopencv_calib3d2413d \
-lopencv_objdetect2413d \
-lopencv_contrib2413d \
-lopencv_legacy2413d \
-lopencv_flann2413d \
-lopencv_ml2413d   \
-lopencv_calib3d2413d  \
-lopencv_gpu2413d  \
-lopencv_ts2413d   \
-lopencv_nonfree2413d  \
-lopencv_ocl2413d  \
-lopencv_photo2413d    \
-lopencv_stitching2413d    \
-lopencv_superres2413d     \
-lopencv_superres2413d

} else : win32:CONFIG(release, debug | release):{
LIBS +=  -LC:/opencv24/build/x64/vc14/lib \
-lopencv_core2413 \
-lopencv_imgproc2413 \
-lopencv_highgui2413 \
-lopencv_ml2413 \
-lopencv_video2413 \
-lopencv_features2d2413 \
-lopencv_calib3d2413 \
-lopencv_objdetect2413 \
-lopencv_contrib2413 \
-lopencv_legacy2413 \
-lopencv_flann2413  \
-lopencv_ml2413   \
-lopencv_calib3d2413  \
-lopencv_gpu2413  \
-lopencv_ts2413   \
-lopencv_nonfree2413  \
-lopencv_ocl2413  \
-lopencv_photo2413    \
-lopencv_stitching2413    \
-lopencv_superres2413     \
-lopencv_superres2413
}

INCLUDEPATH += C:/opencv24/build/include
DEPENDPATH += C:/opencv24/build/include
