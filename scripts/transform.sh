#
set -e

for id in Controlled-GT-Cubic-BBA-LMH Controlled-GT-Cubic-MPC-LMH Controlled-GT-Cubic-BBA-Low
do

    if [[ "$id" == *"LMH"* ]]; then
        f=10
    elif [[ "$id" == *"Low"* ]]; then
        f=3
    fi
    #
    dataset="src/data/datasets/$id"
    fit="logs/fit"
    suffix="$id-gaussian.asym-v$f"
    echo $suffix
    # Auto detect resuming directory from suffix.
    num_detects=0
    datetime=""
    abstract=""
    for detect in $(ls logs/fit); do
        #
        if [[ ${detect} == *${suffix} ]]; then
            #
            echo "Detect: \"${detect}\"."
            datetime=${detect%%:*}
            abstract=${detect##*:}
            num_detects=$((num_detects + 1))
        fi
    done
    if [[ ${num_detects} -ne 1 ]]; then
        #
        echo "Fail to find the ONLY ONE matching for given suffix."
        exit 1
    else
        #
        resume=${fit}/${datetime}:${abstract}
    fi

    #
    seed=42
    device=cpu
    num_random_samples=3

    #
    python3 -u transform.py \
        --suffix "$(basename ${dataset})<=${abstract}" \
        --dataset ${dataset} --transform ${dataset}/full.json \
        --seed ${seed} --device ${device} --jit \
        --resume ${resume} \
        --num-random-samples ${num_random_samples} --num-sample-seconds 300
done
