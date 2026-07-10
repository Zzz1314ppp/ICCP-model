#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=proton_md_final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=g1
#SBATCH --output=md_proton_%j.out
#SBATCH --error=md_proton_%j.err

# Load computing environment modules
module load mkl mpi cuda pwmat2/20251105
export TMPDIR=/dev/shm

##########################################################################
# Configurable Parameters (Modify as required)
##########################################################################
STEP_INTERVAL=50          # Scan proton state every 50 fs
THRESHOLD_DIST=2.0        # Distance threshold for H adsorption on metal (Angstrom)
ADD_PROTON_NUM=1          # Number of H3O+ groups to add after each consumption (1 group = 1O + 3H)
SOLVENT_VACUUM_OFFSET=0.02 # Place H3O oxygen 0.02 fractional z above maximum solvent z
MAX_Z_UPPER_LIMIT=0.95    # Upper z limit for H3O placement to avoid pure vacuum region
PWMAT_PROC=2              # Number of MPI parallel processes for PWmat
INPUT_LIST=("etot.input" "IN.SOLVENT")
MDSTEPS_FILE="MDSTEPS"
MOVEMENT_FILE="MOVEMENT"
O_H_BOND_CUT=1.3          # O-H covalent bond length cutoff (Angstrom)
##########################################################################

# Wait until MD simulation reaches target inspection timestep
wait_md_step() {
    target_step=$1
    while true; do
        [ ! -f "${MDSTEPS_FILE}" ] && sleep 2 && continue
        curr_max_fs=$(awk '/Iter\(fs\)/{print $2}' "${MDSTEPS_FILE}" | sort -g | tail -n1)
        awk -v curr="${curr_max_fs}" -v t="${target_step}" 'BEGIN{exit curr >= t ? 0 : 1}'
        if [ $? -eq 0 ]; then
            echo "==== Reached target ${target_step} MD steps, parse last frame from MOVEMENT ===="
            break
        fi
        sleep 5
    done
}

# Safely terminate PWmat with SIGINT signal and wait for cache data writing
safe_stop_pwmat() {
    pkill -SIGINT PWmat
    sleep 8
    pkill -f PWmat >/dev/null 2>&1
    echo "PWmat process terminated safely, trajectory and wavefunction cache saved completely"
}

# Extract last frame lattice vectors & atomic fractional coordinates into shared memory
parse_last_frame() {
    # Clean old tmp files first
    rm -f /dev/shm/frame_block.tmp /dev/shm/lat.info /dev/shm/o_z.tmp /dev/shm/last_frame.tmp
    # Wait if MOVEMENT not generated
    while [ ! -f "${MOVEMENT_FILE}" ]; do
        echo "MOVEMENT file not generated yet, wait 3s..."
        sleep 3
    done
    # Get last iteration line with frame data
    last_iter_line=$(grep -n "Iteration (fs)" "${MOVEMENT_FILE}" | tail -n1 | cut -d: -f1)
    if [ -z "${last_iter_line}" ]; then
        echo "No Iteration frame found in MOVEMENT, wait 3s..."
        sleep 3
        parse_last_frame
        return
    fi
    sed -n "${last_iter_line},\$p" "${MOVEMENT_FILE}" > /dev/shm/frame_block.tmp

    # Extract three lattice vector rows
    awk '/Lattice vector \(Angstrom\)/{flag=1;next} flag==1{print $1,$2,$3; cnt++; if(cnt==3) flag=0}' /dev/shm/frame_block.tmp > /dev/shm/lat.info
    LAT_X=$(sed -n '1p' /dev/shm/lat.info | awk '{print $1}')
    LAT_Y=$(sed -n '2p' /dev/shm/lat.info | awk '{print $2}')
    LAT_Z=$(sed -n '3p' /dev/shm/lat.info | awk '{print $3}')

    # Fix awk syntax: pass oxygen ID via -v to avoid empty judge error
    awk -v oxy_id="${OXYGEN_TYPE}" '/^[0-9]+[ \t]+[0-9.]+E/{print $1,$2,$3,$4; if($1==oxy_id) print $4 > "/dev/shm/o_z.tmp"}' /dev/shm/frame_block.tmp > /dev/shm/last_frame.tmp
    echo "${LAT_X} ${LAT_Y} ${LAT_Z}" > /dev/shm/lattice_all.info
}

# Automatically calculate fractional z coordinate for placing new H3O at solvent-vacuum boundary
get_solvent_boundary_z() {
    # Wait if oxygen z file missing
    while [ ! -f "/dev/shm/o_z.tmp" ] || [ ! -s "/dev/shm/o_z.tmp" ]; do
        echo "/dev/shm/o_z.tmp empty or missing, reparse frame..."
        parse_last_frame
        sleep 2
    done
    sol_max_fz=$(sort -g /dev/shm/o_z.tmp | tail -n1)
    add_fz=$(awk -v s="${sol_max_fz}" -v off="${SOLVENT_VACUUM_OFFSET}" 'BEGIN{print s + off}')
    final_fz=$(awk -v val="${add_fz}" -v maxu="${MAX_Z_UPPER_LIMIT}" 'BEGIN{print val > maxu ? maxu : val}')
    echo "${final_fz}"
    echo "[Interface Detection] Max solvent fractional z: ${sol_max_fz} | H3O placement z: ${final_fz}" >> proton_monitor.log
}

# Core judgment function: Identify H3O+ and judge whether internal H protons adsorb on metal surface
check_proton_near_metal() {
    parse_last_frame
    read LAT_X LAT_Y LAT_Z < /dev/shm/lattice_all.info
    trigger=0

    rm -f /dev/shm/met.tmp /dev/shm/oxy.tmp /dev/shm/hyd.tmp
    awk -v M="${METAL_ATOM_TYPE}" -v O="${OXYGEN_TYPE}" -v H="${HYDROGEN_TYPE}" '
    {
        atom_id=$1; fx=$2; fy=$3; fz=$4;
        if(atom_id == M) print fx,fy,fz > "/dev/shm/met.tmp"
        if(atom_id == O) print fx,fy,fz > "/dev/shm/oxy.tmp"
        if(atom_id == H) print fx,fy,fz > "/dev/shm/hyd.tmp"
    }' /dev/shm/last_frame.tmp

    # Wait oxygen tmp file exist before reading
    while [ ! -f "/dev/shm/oxy.tmp" ] || [ ! -s "/dev/shm/oxy.tmp" ]; do
        echo "/dev/shm/oxy.tmp missing, reparse frame..."
        parse_last_frame
        sleep 2
    done
    # Traverse every oxygen atom
    while read ox fy fz; do
        oxc=$(awk -v fx="${ox}" -v LX="${LAT_X}" 'BEGIN{print fx * LX}')
        oyc=$(awk -v fy="${fy}" -v LY="${LAT_Y}" 'BEGIN{print fy * LY}')
        ozc=$(awk -v fz="${fz}" -v LZ="${LAT_Z}" 'BEGIN{print fz * LAT_Z}')

        > /dev/shm/bond_h.tmp
        bond_h_num=0
        # Wait hydrogen tmp file exist
        while [ ! -f "/dev/shm/hyd.tmp" ] || [ ! -s "/dev/shm/hyd.tmp" ]; do
            echo "/dev/shm/hyd.tmp missing, reparse frame..."
            parse_last_frame
            sleep 2
        done
        while read hx hy hz; do
            hxc=$(awk -v fx="${hx}" -v LX="${LAT_X}" 'BEGIN{print fx * LX}')
            hyc=$(awk -v fy="${hy}" -v LY="${LAT_Y}" 'BEGIN{print fy * LY}')
            hzc=$(awk -v fz="${hz}" -v LZ="${LAT_Z}" 'BEGIN{print fz * LZ}')
            oh_dist=$(awk -v x1="${oxc}" -v y1="${oyc}" -v z1="${ozc}" -v x2="${hxc}" -v y2="${hyc}" -v z2="${hzc}" 'BEGIN{dx=x1-x2; dy=y1-y2; dz=z1-z2; print sqrt(dx*dx + dy*dy + dz*dz)}')
            awk -v d="${oh_dist}" -v cut="${O_H_BOND_CUT}" 'BEGIN{exit d < cut ? 0 : 1}'
            if [ $? -eq 0 ]; then
                bond_h_num=$((bond_h_num + 1))
                echo "${hx} ${hy} ${hz}" >> /dev/shm/bond_h.tmp
            fi
        done < /dev/shm/hyd.tmp

        # Only process H3O+ (3 bonded H atoms)
        if [ ${bond_h_num} -lt 3 ]; then
            continue
        fi

        # Traverse H3O inner H atoms
        while read hx hy hz; do
            hxc=$(awk -v fx="${hx}" -v LX="${LAT_X}" 'BEGIN{print fx * LX}')
            hyc=$(awk -v fy="${hy}" -v LY="${LAT_Y}" 'BEGIN{print fy * LY}')
            hzc=$(awk -v fz="${hz}" -v LZ="${LAT_Z}" 'BEGIN{print fz * LZ}')
            min_h_m_dist=9999.0

            # Wait metal tmp file exist
            while [ ! -f "/dev/shm/met.tmp" ] || [ ! -s "/dev/shm/met.tmp" ]; do
                echo "/dev/shm/met.tmp missing, reparse frame..."
                parse_last_frame
                sleep 2
            done
            while read mx my mz; do
                mxc=$(awk -v fx="${mx}" -v LX="${LAT_X}" 'BEGIN{print fx * LX}')
                myc=$(awk -v fy="${my}" -v LY="${LAT_Y}" 'BEGIN{print fy * LY}')
                mzc=$(awk -v fz="${mz}" -v LZ="${LAT_Z}" 'BEGIN{print fz * LZ}')
                dist=$(awk -v x1="${hxc}" -v y1="${hyc}" -v z1="${hzc}" -v x2="${mxc}" -v y2="${myc}" -v z2="${mzc}" 'BEGIN{dx=x1-x2; dy=y1-y2; dz=z1-z2; print sqrt(dx*dx + dy*dy + dz*dz)}')
                min_h_m_dist=$(awk -v d="${dist}" -v md="${min_h_m_dist}" 'BEGIN{print d < md ? d : md}')
            done < /dev/shm/met.tmp

            # Trigger adsorption flag
            awk -v d="${min_h_m_dist}" -v thr="${THRESHOLD_DIST}" 'BEGIN{exit d < thr ? 0 : 1}'
            if [ $? -eq 0 ]; then
                trigger=1
                break 3
            fi
        done < /dev/shm/bond_h.tmp
    done < /dev/shm/oxy.tmp

    rm -f /dev/shm/*.tmp
    echo ${trigger}
}

# Generate new atom.config with extra H3O+ at liquid-vacuum boundary
build_new_atom_config() {
    boundary_fz=$(get_solvent_boundary_z)
    read LAT_X LAT_Y LAT_Z < /dev/shm/lattice_all.info
    h_offset=0.018
    rm -f new_atom.config.tmp
    touch new_atom.config.tmp

    echo "pwmat_md_restart_proton_add" >> new_atom.config.tmp
    echo "1.0" >> new_atom.config.tmp
    cat /dev/shm/lat.info >> new_atom.config.tmp
    echo "${ATOM_TYPE_LINE}" >> new_atom.config.tmp

    m_cnt=$(echo "${ATOM_COUNT_LINE}" | awk '{print $1}')
    o_cnt=$(echo "${ATOM_COUNT_LINE}" | awk '{print $2}')
    h_cnt=$(echo "${ATOM_COUNT_LINE}" | awk '{print $3}')

    new_o=$(echo "$o_cnt $ADD_PROTON_NUM" | awk '{print $1 + $2}')
    new_h=$(echo "$h_cnt $ADD_PROTON_NUM" | awk '{print $1 + $2 * 3}')
    new_count_str="${m_cnt} ${new_o} ${new_h}"
    echo "${new_count_str}" >> new_atom.config.tmp

    echo "Selective dynamics" >> new_atom.config.tmp
    awk '{print $1, $2, $3, $4, 1, 1, 1}' /dev/shm/last_frame.tmp >> new_atom.config.tmp

    for (( g=0; g<ADD_PROTON_NUM; g++ )); do
        off_x=$(awk -v g="${g}" 'BEGIN{print 0.5 + g * 0.08}')
        off_y=$(awk -v g="${g}" 'BEGIN{print 0.5 - g * 0.08}')
        h_z_pos=$(awk -v b="${boundary_fz}" -v ho="${h_offset}" 'BEGIN{print b + ho}')
        ox_h=$(awk -v x="${off_x}" -v ho="${h_offset}" 'BEGIN{print x + ho}')
        ox_l=$(awk -v x="${off_x}" -v ho="${h_offset}" 'BEGIN{print x - ho}')
        oy_h=$(awk -v y="${off_y}" -v ho="${h_offset}" 'BEGIN{print y + ho}')

        echo "${OXYGEN_TYPE}    ${off_x}    ${off_y}    ${boundary_fz}    1 1 1" >> new_atom.config.tmp
        echo "${HYDROGEN_TYPE}    ${ox_h}    ${off_y}    ${h_z_pos}    1 1 1" >> new_atom.config.tmp
        echo "${HYDROGEN_TYPE}    ${ox_l}    ${off_y}    ${h_z_pos}    1 1 1" >> new_atom.config.tmp
        echo "${HYDROGEN_TYPE}    ${off_x}    ${oy_h}    ${h_z_pos}    1 1 1" >> new_atom.config.tmp
    done
    mv new_atom.config.tmp atom.config
}

# Read & cache metal/O/H atom ID config on first run
read_atom_element_config() {
    echo "==================== System Atom Type Configuration ===================="
    awk '/^[0-9]+[ \t]+[0-9.]+/{print $1}' atom.config | sort -nu > .tmp_atom_list
    echo "All atom IDs in current system:"
    cat .tmp_atom_list
    read -p "Input metal atom ID (single number e.g.78): " METAL_ATOM_TYPE
    read -p "Input oxygen atom ID (e.g.8): " OXYGEN_TYPE
    read -p "Input hydrogen atom ID (e.g.1): " HYDROGEN_TYPE
    echo "${METAL_ATOM_TYPE} ${OXYGEN_TYPE} ${HYDROGEN_TYPE}" > .element_cache
    rm -f .tmp_atom_list
    echo "Atom config cached to .element_cache, no re-input required next run"
    echo "==== PWmat proton adsorption monitor log ====" > proton_monitor.log
}

# Main persistent PWmat MD loop, launch PWmat only once
pwmat_md_main() {
    # Check required input files
    pre_check_input
    if [ ! -f .element_cache ]; then
        read_atom_element_config
    else
        read METAL_ATOM_TYPE OXYGEN_TYPE HYDROGEN_TYPE < .element_cache
        echo "Load cached metal/O/H atom ID configuration" >> proton_monitor.log
    fi

    step_count=0
    echo "==== Launch single persistent PWmat MD process ONCE ===="
    mpirun -np ${PWMAT_PROC} PWmat > output 2> pwmat.err &
    PW_PID=$!
    echo "PWmat background PID = ${PW_PID}"
    echo "Wait 8s for PWmat initialization and MDSTEPS generation..."
    sleep 8

    while true; do
        curr_target_step=$(( step_count * STEP_INTERVAL ))
        echo "=================================================="
        echo "Round ${step_count}, target inspection step: ${curr_target_step}"
        echo "=================================================="
        wait_md_step ${curr_target_step}
        trigger_flag=$(check_proton_near_metal)

        if [ ${trigger_flag} -eq 1 ]; then
            echo "Detected H3O proton H close to metal surface, proton adsorption occurred!"
            echo "Round ${step_count}: Stop PWmat and submit new job with extra H3O+" >> proton_monitor.log
            safe_stop_pwmat
            build_new_atom_config
            # Create independent restart folder
            restart_dir="pwmat_restart_$(date +%Y%m%d_%H%M%S)"
            mkdir -p ${restart_dir}
            for f in "${INPUT_LIST[@]}"; do cp ${f} ${restart_dir}/; done
            mv atom.config ${restart_dir}/
            cp $0 .element_cache proton_monitor.log ${restart_dir}/
            cd ${restart_dir}
            sbatch $0
            echo "New proton-supplemented PWmat MD task submitted, exit current run"
            exit 0
        fi
        echo "No proton adsorption detected, continue MD without restart PWmat"
        step_count=$(( step_count + 1 ))
        sleep 10
    done
}

# Pre-check all mandatory input files
pre_check_input() {
    echo "==== Input file integrity pre-check start ===="
    for f in "${INPUT_LIST[@]}" "atom.config"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: Missing mandatory input file $f" >&2
            exit 1
        fi
    done
    echo "All required input files exist"
}

# Global script entry
echo "=== TASK START TIMESTAMP: $(date) ==="
START_TIME=$(date +%s)

# PWmat executable validate
PWMAT_EXE="PWmat"
if [ ! -x "$(command -v ${PWMAT_EXE})" ]; then
    echo "ERROR: PWmat executable not found in PATH" >&2
    INFO_DIR="$HOME/pwmat_information"
    mkdir -p "$INFO_DIR"
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))
    record_job "FAILED(PWMAT_EXE_MISSING)" $END_TIME $RUNTIME
    exit 1
fi

# Execute main MD monitoring logic
pwmat_md_main

# Capture script exit state
PW_STATUS=$?
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# Final completion banner
echo ""
echo "##########################################################"
echo "###               PWmat MD SIMULATION FINISHED          ###"
[ $PW_STATUS -eq 0 ] && echo "###                   Status: SUCCESS                    ###" || echo "###                   Status: FAILED                     ###"
echo "### Job ${SLURM_JOB_ID} running on host: $(hostname)"
echo "##########################################################"
echo "Completion time: $(date)"
echo "Working directory: $PWD"
echo "Job metadata archive path: $INFO_DIR"
echo ""

# Save permanent job record
INFO_DIR="$HOME/pwmat_information"
mkdir -p "$INFO_DIR"
record_job() {
    local status=$1
    local end_time=$2
    local runtime=$3
    runtime_str=$(printf "%d:%02d:%02d" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60)))
    JOB_INFO="$INFO_DIR/job_${SLURM_JOB_ID}.info"
    cat > "$JOB_INFO" <<EOF
=== PWmat MD JOB METADATA ===
Job ID:       $SLURM_JOB_ID
Status:       $status
Start time:   $(date -d @$START_TIME)
End time:     $(date -d @$end_time)
Walltime:     $runtime_str
Work dir:     $PWD
Script file:  $0
Log files:
  Slurm out: md_proton_${SLURM_JOB_ID}.out
  PWmat log: output / pwmat.err
=== FULL SCRIPT CONTENT ===
$(cat $0)
EOF
    echo "$(date -d @$end_time) | JobID:$SLURM_JOB_ID | $status | Walltime:$runtime_str | Dir:$PWD" >> "$INFO_DIR/jobs.index"
    echo "Complete job record saved to $JOB_INFO"
}
if [ $PW_STATUS -eq 0 ]; then
    record_job "SUCCESS" $END_TIME $RUNTIME
else
    record_job "FAILED($PW_STATUS)" $END_TIME $RUNTIME
fi
exit $PW_STATUS
