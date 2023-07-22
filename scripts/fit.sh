#
set -e


for id in Controlled-GT-Cubic-BBA-LMH Controlled-GT-Cubic-MPC-LMH Controlled-GT-Cubic-BBA-Low
do
    if [[ "$id" == *"LMH"* ]]; then
        f=10
        varinit=0.0625
        varmax_rest=0.25 # Expected variance (not standard deviation) maximum.
        varmax_head=1
    elif [[ "$id" == *"$Low"* ]]; then
        f=3
        varinit=1
        varmax_rest=4 # Expected variance (not standard deviation) maximum.
        varmax_head=2.25
    fi

    dataset="src/data/datasets/$id"

    capacity_max=8.0
    capacity_unit=0.5
    
    head_by_time=5.0 # Heading variance time.
    head_by_chunk=5 # Heading variance time.

    #
    seed=42
    device=cpu
    initial=generic
    num_epochs=25
    transition_unit=5.0
    vareta=0.0001

    # Learn with arbitrary prior knowledge (uniform initial and asymmetric Gaussian transition).
    transition=gaussian.asym
    emission=v$f
    initeta=0 #not learn from initial.
    transeta=0.1

    #Initial trans matrix is uniform {1/#of states}.

    python3 -u fit.py \
        --suffix $(basename ${dataset})-${transition}-${emission} \
        --dataset ${dataset} --train ${dataset}/full.json --valid ${dataset}/full.json --test ${dataset}/full.json \
        --seed ${seed} --device ${device} --jit \
        --initial ${initial} --transition ${transition} --emission ${emission} \
        --num-epochs ${num_epochs} \
        --capacity-max ${capacity_max} --capacity-unit ${capacity_unit} \
        --transition-unit ${transition_unit} \
        --initeta ${initeta} --transeta ${transeta} --vareta ${vareta} --varinit ${varinit} --varmax-head ${varmax_head} \
        --varmax-rest ${varmax_rest} --head-by-time ${head_by_time} --head-by-chunk ${head_by_chunk} \
        --transextra 5 --include-beyond --smooth 0.05
done
