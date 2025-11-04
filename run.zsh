#!/bin/zsh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${SCRIPT_DIR}/dolbom_venv"

find_python() {
    local candidates=("python3.10" "python3.11" "python3")
    local candidate version

    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1; then
            version="$("$candidate" -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')"
            if [[ "$version" == "3.10" || "$version" == "3.11" ]]; then
                echo "$candidate"
                return 0
            fi
        fi
    done

    return 1
}

echo "=== [1] 운영체제 감지 중... ==="
OS="$(uname -s)"
DISTRO=""

echo "Detected OS: $OS"

if [[ "$OS" == "Linux" ]]; then
    if grep -qi microsoft /proc/version; then
        echo "Running under WSL (Ubuntu assumed)..."
        DISTRO="ubuntu"
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
    fi

    if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
        echo "Setting up for Ubuntu/Debian..."
        sudo apt update
        sudo apt install software-properties-common -y
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt update
        sudo apt install python3.10 python3.10-venv python3.10-dev -y
        sudo apt-get install -y ffmpeg build-essential
        sudo apt-get install -y portaudio19-dev libasound-dev
        sudo apt-get install -y libxcb-xinerama0 libxcb1 libxcb-util1 libx11-xcb1 libxrender1 libxi6 libxext6
        sudo apt-get install -y qtbase5-dev qttools5-dev-tools qt5-qmake
        sudo apt-get install -y python3-gi python3-gi-cairo gir1.2-gtk-4.0
        sudo apt-get install -y libgirepository-2.0-dev gcc libcairo2-dev pkg-config python3-dev python3-venv
        sudo apt-get install fonts-nanum


    elif [[ "$DISTRO" == "arch" || "$DISTRO" == "manjaro" ]]; then
        echo "Setting up for Arch/Manjaro..."
        sudo pacman -Syu --noconfirm
        sudo pacman -S --noconfirm ffmpeg base-devel portaudio libasound
        sudo pacman -S --noconfirm cairo pkgconf python-gobject gtk4 python-virtualenv qt5-base

    elif [[ "$DISTRO" == "centos" || "$DISTRO" == "rhel" ]]; then
        echo "Setting up for CentOS/RHEL..."
        sudo yum install -y epel-release
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y ffmpeg gcc-c++ make portaudio-devel alsa-lib-devel cairo-devel pkgconfig python3-devel python3-venv gobject-introspection-devel gtk4
        sudo dnf install gcc openssl-devel bzip2-devel libffi-devel
        cd /usr/src
        sudo curl -O https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
        sudo tar xzf Python-3.10.13.tgz
        cd Python-3.10.13
        sudo ./configure --enable-optimizations
        sudo make altinstall  # python3.10 명령어 생김

    else
        echo "Unsupported Linux distro: $DISTRO"
        exit 1
    fi

elif [[ "$OS" == "Darwin" ]]; then
    echo "Setting up for macOS..."
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install ffmpeg portaudio cairo pkg-config python gobject-introspection gtk4 python@3.10 qt@5
    if [[ -d "/opt/homebrew/opt/python@3.10/bin" ]]; then
        export PATH="/opt/homebrew/opt/python@3.10/bin:$PATH"
    fi
    if [[ -d "/opt/homebrew/opt/qt@5/bin" ]]; then
        export PATH="/opt/homebrew/opt/qt@5/bin:$PATH"
    fi

else
    echo "Unsupported OS: $OS"
    exit 1
fi

cd "$SCRIPT_DIR"

if ! PYTHON_BIN="$(find_python)"; then
    echo "Python 3.10 또는 3.11을 찾을 수 없습니다. 먼저 설치한 뒤 다시 실행해주세요."
    exit 1
fi
PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')"
echo "Using Python interpreter: $PYTHON_BIN (version $PYTHON_VERSION)"



echo "=== [2] 가상환경 생성 중... ==="

echo "=== 실행 중인 가상환경 비활성화 하기... ==="
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate || true
fi
if command -v conda >/dev/null 2>&1; then
    conda deactivate || true
fi

echo "=== 가상환경 생성... ==="
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "가상환경 'dolbom_venv' 생성 완료"
else
    echo "가상환경 'dolbom_venv' 이미 존재함"
fi

source "$VENV_DIR/bin/activate"
VENV_PYTHON="$VENV_DIR/bin/python"

echo "=== 더 이상 사용되지 않는 이전 버전의 패키지 제거 중... ==="
rm -rf dora_venv
"$VENV_PYTHON" -m pip uninstall -y gpt4all

echo "=== [3] 기존 충돌 패키지 제거 중... ==="
"$VENV_PYTHON" -m pip uninstall -y PyQt5 opencv-python-headless opencv-python opencv-contrib-python

echo "=== [4] Python 패키지 설치 중... ==="
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -r requirements.txt

echo "=== [5] 데이터베이스 초기화 중... ==="
if [[ -f ".env" ]]; then
    echo ".env 파일 로드..."
    set -a
    source .env
    set +a

    if ! command -v mysql >/dev/null 2>&1; then
        echo "mysql 클라이언트를 찾을 수 없습니다. 설치 후 다시 실행해주세요."
        exit 1
    fi

    if [[ -z "${DB_NAME}" ]]; then
        echo "DB_NAME 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요."
        exit 1
    fi

    mysql_args=(-h "${DB_HOST:-localhost}" -P "${DB_PORT:-3306}" -u "${DB_USER:-root}")

    if [[ -n "${DB_PASSWORD}" ]]; then
        export MYSQL_PWD="${DB_PASSWORD}"
    else
        unset MYSQL_PWD
    fi

    mysql "${mysql_args[@]}" -e "CREATE DATABASE IF NOT EXISTS \`${DB_NAME}\` CHARACTER SET utf8mb4;"
    mysql "${mysql_args[@]}" "${DB_NAME}" < db/dora.sql
    unset MYSQL_PWD

    echo "=== [6] 샘플 데이터 삽입 중... ==="
    "$VENV_PYTHON" -m db.seed
else
    echo ".env 파일을 찾을 수 없어 DB 초기화를 건너뜁니다."
fi

echo ""
echo "모든 설치가 완료되었습니다!"
echo ""
echo "  ▶ 가상환경 재진입:        source dolbom_venv/bin/activate"
echo "  ▶ 가상환경 종료:          deactivate"
echo ""
echo "  ▶ 기본 실행:              python3 main.py"
echo "  ▶ GUI 실행:              python3 main.py      # '-g' 없이도 실행됩니다"
echo "  ▶ GUI 종료 방법:         창 닫기 또는 Ctrl + C"
echo "  ▶ GUI 강제 중지:         python3 main.py --stop"
echo ""

read -r "run_now?▶ 설치가 완료되었습니다. 지금 바로 실행할까요? (y/N): "

if [[ "$run_now" == "y" || "$run_now" == "Y" || "$run_now" == "yes" || "$run_now" == "네" || "$run_now" == "ㅇ" || "$run_now" == "ㅇㅇ" ]]; then
    echo ""
    echo "main.py 실행 중..."
    "$VENV_PYTHON" main.py
else
    echo "실행을 건너뜁니다. 필요 시 다음 명령을 입력하세요:"
    echo "   source dolbom_venv/bin/activate"
    echo "   python3 main.py"
fi
