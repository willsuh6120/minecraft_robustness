#!/bin/bash
set -euo pipefail

replaceable=0
port=0
seed="NONE"
maxMem="${MINESTUDIO_JAVA_MAX_MEM:-2G}"
device="egl"
fatjar=build/libs/mcprec-6.13.jar
java_bin="${MINERL_JAVA_BIN:-java}"

while [ $# -gt 0 ]
do
    case "$1" in
        -replaceable) replaceable=1;;
        -port) port="$2"; shift;;
        -seed) seed="$2"; shift;;
        -maxMem) maxMem="$2"; shift;;
        -device) device="$2"; shift;;
        -fatjar) fatjar="$2"; shift;;
        *) echo >&2 \
            "usage: $0 [-replaceable] [-port <port>] [-seed <seed>] [-maxMem <maxMem>] [-device <device>] [-fatjar <fatjar>]"
            exit 1;;
    esac
    shift
done

if ! [[ $port =~ ^-?[0-9]+$ ]]; then
    echo "Port value should be numeric"
    exit 1
fi


if [ \( $port -lt 0 \) -o \( $port -gt 65535 \) ]; then
    echo "Port value out of range 0-65535"
    exit 1
fi

mkdir -p logs crash-reports resourcepacks shaderpacks saves
touch logs/latest.log
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$PWD/.xdg_runtime}"
mkdir -p "$XDG_RUNTIME_DIR"

if [ "$device" == "cpu" ]; then
    if [ "$(uname)" = "Darwin" ]; then
        "$java_bin" -Dlog4j2.disable.jmx=true -Xmx$maxMem -XstartOnFirstThread -jar "$fatjar" --envPort="$port"
    elif [ -n "${DISPLAY:-}" ]; then
        "$java_bin" -Dlog4j2.disable.jmx=true -Xmx$maxMem -jar "$fatjar" --envPort="$port"
    else
        xvfb-run -a "$java_bin" -Dlog4j2.disable.jmx=true -Xmx$maxMem -jar "$fatjar" --envPort="$port"
    fi
else
    vglrun -d "$device" "$java_bin" -Dlog4j2.disable.jmx=true -Xmx$maxMem -jar "$fatjar" --envPort="$port"
fi

exit 0
