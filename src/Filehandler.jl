module FileHandler

using Dates
using DelimitedFiles
using Printf
using GZip

export FileManager
export create_cfd_instance, deconstruct_cfd_instance, execute_external_file, retrieve_value, init_folder, infer_expression_to_file, get_next_id


struct FileManager
    start_date::String
    target_name::String
    running_template_path::String
    input_template_name::String
    eval_template_path::String
    output_source::String
    cost_fun_identifier::Union{String,Vector{String}}
    eval_model_file::String
    task_folder_suffix::String
    config::Dict{String,Any}
    id_counter::Threads.Atomic{Int}

    function FileManager(config::Dict{String,Any})
        start_date = Dates.format(now(), "ddHHMMSS")
        target_name = get(config, "name", "cfd_loop") * "_" * start_date
        running_template_path = get(config, "running_template_path", "running_template")
        input_template_name = get(config, "input_template_name", "input_template")
        eval_template_path = get(config, "eval_template_path", "eval_template")
        output_source = get(config, "output_folder_name", "output")
        cost_fun_identifier = get(config, "target_path", ["C1"])
        eval_model_file = get(config, "eval_model_file", "eval_model.py")
        task_folder_suffix = get(config, "suffix", "*_eve_task")
        id_counter = Threads.Atomic{Int}(0)
        fm = new(start_date, target_name, running_template_path, input_template_name,
            eval_template_path, output_source, cost_fun_identifier,
            eval_model_file, task_folder_suffix, config, id_counter)

        init_folder(fm)
        return fm
    end
end

function get_next_id(fm::FileManager)
    return Threads.atomic_add!(fm.id_counter, 1)
end

function create_cfd_instance(fm::FileManager, ind_id::Int, ind_expression::Vector{String})
    indi_name = "run_$ind_id"
    indi_operating_path = joinpath(fm.target_name, indi_name)
    cp(joinpath(fm.target_name, fm.eval_template_path), indi_operating_path, force=true)
    infer_expression_to_file(fm, indi_operating_path, ind_expression, ind_id)
    return indi_operating_path
end

function deconstruct_cfd_instance(fm::FileManager, target_path::String)
    rm(target_path, recursive=true, force=true)
end

function execute_external_file(fm::FileManager, target_path::String; python="python")
    run(`$python $(joinpath(target_path, fm.eval_model_file))`)
end

function retrieve_value(fm::FileManager, target_path::String)
    errs = Float64[]
    for elem in fm.cost_fun_identifier
        file_path = joinpath(target_path, fm.output_source, "$elem.edf.gz")
        if isfile(file_path)
            try
                GZip.open(file_path, "r") do io
                    data = readdlm(io, Float64)
                    if !isempty(data)
                        push!(errs, data[1])
                    else
                        @warn "Empty data file for $elem"
                        push!(errs, NaN)
                    end
                end
            catch e
                @warn "Error reading file $file_path: $e"
                push!(errs, NaN)
            end
        else
            @warn "File not found: $file_path"
            push!(errs, NaN)
        end
    end
    return Tuple(errs)
end

function init_folder(fm::FileManager)
    cp(fm.running_template_path, fm.target_name, force=true)
    println("Run is created in $(fm.target_name)")
end

function infer_expression_to_file(fm::FileManager, operating_path::String, ind_expression::Vector{String}, ind_id::Int)
    input_path_temp = joinpath(operating_path, fm.input_template_name)
    open(input_path_temp, "a") do f
        println(f, join(ind_expression, ","))
    end
    mv(input_path_temp, joinpath(operating_path, "input_$ind_id"), force=true)
end




end
