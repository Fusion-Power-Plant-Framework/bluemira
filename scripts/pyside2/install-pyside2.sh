set -e

NJOBS=$(nproc --ignore=2)

while getopts j: option
do
  case "${option}"
  in
    j) NJOBS=${OPTARG};;
  esac
done

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

cd pyside-setup
SKIP_MODS=Multimedia,MultimediaWidgets,Positioning,Location,Qml,Quick,\
QuickWidgets,RemoteObjects,Scxml,Script,ScriptTools,Sensors,SerialPort,TextToSpeech,\
Charts,DataVisualization,WebChannel,WebEngineCore,WebEngine,WebEngineWidgets,WebSockets,\
3DCore,3DRender,3DInput,3DLogic,3DAnimation,3DExtras
# https://bugreports.qt.io/browse/PYSIDE-1873 (limited api switch)
python3 setup.py build --qmake=/usr/local/Qt-5.15.5/bin/qmake --parallel=$NJOBS --limited-api=yes --skip-modules=$SKIP_MODS
python3 setup.py install --qmake=/usr/local/Qt-5.15.5/bin/qmake --parallel=$NJOBS --limited-api=yes
