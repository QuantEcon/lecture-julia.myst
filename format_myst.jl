# Utility to run JuliaFormatter.jl on the julia code within myst markdown files
# Written for CLI acccess:  for example, call it with
#  julia format_myst.jl lectures/getting_started_julia/getting_started.md

# Test with the following examples:
# format_myst("lectures/getting_started_julia/getting_started.md", "lectures/getting_started_julia/getting_started2.md")
# format_myst!("lectures/getting_started_julia/getting_started.md")
using JuliaFormatter

# Substrings with unicode don't work very well
substring(str, start, stop) = str[nextind(str, 0, start):nextind(str, 0, stop)]

# inplace modification
function format_myst!(input_file_path, extra_replacements = false,
                      ignore_errors = false)
    format_myst(input_file_path, input_file_path, extra_replacements, ignore_errors)
end
function format_myst(input_file_path, output_file_path, extra_replacements = false,
                     ignore_errors = false)
    # It has one capturing group which corresponds to the code inside the block
    code_block_pattern = r"(```{code-cell} julia\n)([\s\S]*?)(\n```)"

    # Extract config files/etc.  Requires a .JuliaFormatter.toml file
    config_file = JuliaFormatter.find_config_file(".")
    style = pop!(config_file, "style")
    config = Dict(Symbol(a) => b for (a, b) in config_file)

    function format_code_block(m)
        try
            # m is the whole match, including the start and end markers    
            block_length = length(m)
            start_marker = substring(m, 1, 21) # length of ```{code-cell} julia\n
            code = String(substring(m, 22, block_length - 4))
            end_marker = substring(m, block_length - 3, block_length)

            # don't transform if it has myst tags
            if substring(code, 1, min(length(code), 3)) == "---"
                return m
            end

            # Use Julia Formatter and the loaded style file
            transformed_code = format_text(code, style; config...)

            # Return the new block
            return start_marker * transformed_code * end_marker
        catch e
            println("Failed to format code block:")
            print(m)
            if ignore_errors
                return m # i.e., don't change anything
            else
                throw(e) 
            end
        end
    end

    # Additional replacements are optional.  This may be useful when replacing variable names to make it easier to type in ascii
    replacements = Dict("α" => "alpha", "β" => "beta", "γ" => "gamma", "≤" => "<=",
    "≥" => ">=", "Σ" => "Sigma", "σ" => "sigma","μ"=>"mu","ϕ"=>"phi","ψ"=>"psi","ϵ"=>"epsilon",
    "δ"=>"delta","θ" => "theta","ζ"=>"zeta","X̄" => "X_bar","p̄" => "p_bar","x̂" => "x_hat","λ"=>"lambda",
    "ρ"=>"rho","u′" => "u_prime" , "f′"=>"f_prime"," ∂u∂c"=>"dudc","Π"=>"Pi","π"=>"pi"," ξ"=>"Xi","c̄"=>"c_bar","w̄"=>"w_bar") 

    # Replace the code blocks in the content and handle exceptions
    try
        # Read in the file
        file_content = read(input_file_path, String)

        new_content = replace(file_content,
                              code_block_pattern => format_code_block)

        if extra_replacements # optional
            new_content = replace(new_content, replacements...)
        end

        open(output_file_path, "w") do f
            write(f, new_content)
        end
        return true
    catch e
        println("Failed to process the markdown file at $input_file_path")
        print(e)
        return false
    end
end

if length(ARGS) == 0
    println("Please provide a file path to a myst markdown file")
    exit(1)
else
    use_extra_replacements = length(ARGS) > 1 && ARGS[2] == "true"
    print("Replacing file at $(ARGS[1]) with formatted version.  Additional replacements = $use_extra_replacements\n")
    success = format_myst!(ARGS[1],use_extra_replacements)
    success || exit(1)
end