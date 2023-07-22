#
set -e

# Using package installer for Python to maintain.
install() {
    #
    local name
    local extra
    local version

    #
    name=${1}
    extra=${2}
    version=${3}
    shift 3

    #
    if [[ ${#extra} -eq 0 ]]; then
        #
        echo "123"
        pip3 install --no-cache-dir --upgrade ${name}==${version} ${*}
    else
        #
        echo "2345"
        pip3 install --no-cache-dir --upgrade ${name}[${extra}]==${version} ${*}
    fi
    installed[${name}]=${version}
}

#
thnum="1.13.0"
cunum="cu116"

# @align[begin]
# <$1 <$2 <$3 >$4 $*
# @align[end]
install pandas "" 1.5.3
install torch "" ${thnum} --extra-index-url https://download.pytorch.org/whl/${cunum}
install scikit-learn "" 1.1.3
install seaborn "" 0.12.1
install more_itertools "" 8.14.0
install mypy "" 0.982
install pytest "" 7.2.0
install pytest-cov "" 4.0.0
install black "" 22.10.0
install hmmlearn "" 0.2.8

#
outdate() {
    #
    local nlns
    local name
    local latest

    #
    latests=()
    nlns=0
    while IFS= read -r line; do
        #
        nlns=$((nlns + 1))
        [[ ${nlns} -gt 2 ]] || continue

        #
        name=$(echo ${line} | awk "{print \$1}")
        latest=$(echo ${line} | awk "{print \$3}")
        latests[${name}]=${latest}
    done <<<$(pip list --outdated)
}

#
outdate
for package in ${!installed[@]}; do
    #
    if [[ -n ${latests[${package}]} ]]; then
        #
        msg1="\x1b[1;93m${package}\x1b[0m"
        msg2="\x1b[94m${installed[${package}]}\x1b[0m"
        msg3="${msg1} (${msg2}) is \x1b[4;93moutdated\x1b[0m"
        msg4="latest version is \x1b[94m${latests[${package}]}\x1b[0m"
        echo -e "${msg3} (${msg4})."
    fi
done
