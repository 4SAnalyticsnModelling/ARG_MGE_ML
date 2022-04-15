# This script evaluates relative feature importance based on shapley values for predicted abundance of ARG and MGE from tuned and trained best models among the trained RF (Random Forest), XGBOOST(Extreme Gradient Boost), and NNET(Artificial Neural Network) regression models
# Install required packages
using Pkg;
Pkg.add(["DataFrames", "CSV", "Statistics", "BSON", "MLJ", "MLJXGBoostInterface", "MLJDecisionTreeInterface", "Flux", "Flux.Zygote"]);
Pkg.add("https://github.com/4SAnalyticsnModelling/MLJulFluxFun"); #This is a module which was developed as a part of this project. All the functions used below for model training are imported from the MLJulFluxFun module.
Pkg.add("https://github.com/4SAnalyticsnModelling/ShapML.jl");
# Import the required packages
using DataFrames, CSV, Statistics, BSON, MLJ, MLJXGBoostInterface, MLJDecisionTreeInterface, Flux, Flux.Zygote, MLJulFluxFun, ShapML;
# Set current directory as working directory
cd(@__DIR__);
# Predefined functions
function jitter(a, width_jit)
    a = convert.(Float64, a)
    a .+= rand(size(a)...) .* width_jit .- width_jit .* 0.5
    return a
end
function predict_function(model, data :: DataFrame)
  data_pred = DataFrame(y_pred = vec(model(Array(data)')))
  return data_pred
end
function normalize(x)
    return (x .- extrema(x)[1])/(extrema(x)[2] - extrema(x)[1])
end
# Import and process data for relative feature importance analyses in predicting ARG abundance
arg_mge_df = DataFrame(CSV.File("../../assets/arg_mge_data_for_ML.csv", normalizenames = true));
arg_df0 = arg_mge_df[!, [:temperature, :residence_time, :operation, :feedstock_and_co_substrate, :additives_or_chemicals, :pre_treatment, :arg_removal_efficiency]];
arg_df0[!, :operation] .= ifelse.(arg_df0.operation .== "Batch", 0.0, 1.0);
arg_df0[!, :additives_or_chemicals] .= ifelse.(arg_df0.additives_or_chemicals .== "No", 0.0, 1.0);
arg_df0[!, :pre_treatment] .= ifelse.(arg_df0.pre_treatment .== "No", 0.0, 1.0);
list_cols = [:feedstock_and_co_substrate]
for items in list_cols
    arg_df0[!, items] .= MLJ.categorical(arg_df0[!, items])
end
hot = MLJ.OneHotEncoder(ordered_factor = true);
mach = MLJ.fit!(machine(hot, arg_df0), verbosity = 0);
arg_df = MLJ.transform(mach, arg_df0);
show(first(arg_df, 6), allcols = true)
# Import tuned and trained NNET Model and performed prediction on ARG abundance
x = arg_df[:, 1:13];
params_of_interest = "ARG";
model_name = "nnet"
J, N = size(x);
mutable struct nnet_mod_builder
    n1 :: Int
    n2 :: Int
    n3 :: Int
end
function nnet_build(nn :: nnet_mod_builder, n_in, n_out)
    return Flux.Chain(Dense(n_in, nn.n1, relu, init = Flux.kaiming_normal),
                 Dense(nn.n1, nn.n2, relu, init = Flux.kaiming_normal),
                 Dense(nn.n2, nn.n3, relu, init = Flux.kaiming_normal),
                 Dense(nn.n3, n_out, init = Flux.kaiming_normal))
end
nnet_model = nnet_build(nnet_mod_builder(N * 6, N * 4, N * 2), N, 1);
target_location = params_of_interest * "_" * model_name * "_trained_model";
trained_model = readdir(target_location);
trained_model = trained_model[contains.(trained_model, ".bson")][2];
nnet_trained_model_weights = BSON.load(target_location .* "/" .* trained_model);
Flux.loadparams!(nnet_model, nnet_trained_model_weights[:weights]);
trained_scaler = readdir(target_location);
trained_scaler = trained_scaler[contains.(trained_scaler, ".bson")][1];
trained_scaler = BSON.load(target_location .* "/" .* trained_scaler);
col_names = Symbol.(names(x));
x = MLJulFluxFun.scale_transform(trained_scaler[:x_scaler], Matrix(x));
x = DataFrame(x, col_names);
# Perform feature importance analyses in predicting ARG abundance using shapley values
shap_df = ShapML.shap(explain = x,
                        reference = x,
                        model = nnet_model,
                        predict_function = predict_function,
                        sample_size = 60,
                        seed = 1);
shap_df_mean = combine(groupby(shap_df, [:feature_name]),
:shap_effect => (x -> mean(abs.(x))) => :mean_shap_values);
sort!(shap_df_mean, [:mean_shap_values], rev = true);
shap_df_mean[!, :feature_tag] .= length(shap_df_mean.feature_name):-1:1;
shap_df = innerjoin(shap_df_mean, shap_df, on = :feature_name);
shap_df[!, :shap_flag] .= ifelse.(shap_df.shap_effect .> 0.0, 1, ifelse.(shap_df.shap_effect .< 0.0, 2, 3));
shap_df1 = combine(groupby(shap_df, [:feature_name]),
:feature_value => (x -> normalize(x)) => :feature_value_normalized,
:shap_effect => :shap_effect,
:feature_tag => :feature_tag);
show(first(shap_df1, 6), allcols = true)
feat_label_df = DataFrame(feature_name = unique(shap_df.feature_name),
feat_labels = ["Temperature (°C)", "Residence Time (days)", "Operation\n(Batch = 0, Semi-continous = 1)", "Feedstock and Co-substrate:\nFood Waste\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nFood Waste + Manure\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nFood Waste + Oil\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nFood Waste + Sewage Sludge\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nManure\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nManure + Wheat Straw\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nPharmaceutical Waste Sludge\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nSewage Sludge\n(No = 0, Yes = 1)", "Additives/Chemicals\n(No = 0, Yes = 1)",
"Pre-treatment\n(No = 0, Yes = 1)"]);
shap_df1 = innerjoin(feat_label_df, shap_df1, on = :feature_name);
shap_df_mean = innerjoin(feat_label_df, shap_df_mean, on = :feature_name);

# Import and process data for relative feature importance analyses in predicting MGE abundance
mge_df0 = arg_mge_df[!, [:temperature, :residence_time, :operation, :feedstock_and_co_substrate, :additives_or_chemicals, :pre_treatment, :mge_removal_efficiency]];
mge_df0 = mge_df0[mge_df0[!, :mge_removal_efficiency] .< 4000, :];
mge_df0[!, :operation] .= ifelse.(mge_df0.operation .== "Batch", 0.0, 1.0);
mge_df0[!, :additives_or_chemicals] .= ifelse.(mge_df0.additives_or_chemicals .== "No", 0.0, 1.0);
mge_df0[!, :pre_treatment] .= ifelse.(mge_df0.pre_treatment .== "No", 0.0, 1.0);
list_cols = [:feedstock_and_co_substrate]
for items in list_cols
    mge_df0[!, items] .= MLJ.categorical(mge_df0[!, items])
end
hot = MLJ.OneHotEncoder(ordered_factor = true);
mach = MLJ.fit!(machine(hot, mge_df0), verbosity = 0);
mge_df = MLJ.transform(mach, mge_df0);
show(first(mge_df, 6), allcols = true)
# Import tuned and trained NNET Model and performed prediction on MGE abundance
x = mge_df[:, 1:13];
params_of_interest = "MGE";
model_name = "nnet"
J, N = size(x);
mutable struct nnet_mod_builder
    n1 :: Int
    n2 :: Int
    n3 :: Int
end
function nnet_build(nn :: nnet_mod_builder, n_in, n_out)
    return Flux.Chain(Dense(n_in, nn.n1, relu, init = Flux.kaiming_normal),
                 Dense(nn.n1, nn.n2, relu, init = Flux.kaiming_normal),
                 Dense(nn.n2, nn.n3, relu, init = Flux.kaiming_normal),
                 Dense(nn.n3, n_out, init = Flux.kaiming_normal))
end
nnet_model = nnet_build(nnet_mod_builder(N * 6, N * 4, N * 2), N, 1);
target_location = params_of_interest * "_" * model_name * "_trained_model";
trained_model = readdir(target_location);
trained_model = trained_model[contains.(trained_model, ".bson")][2];
nnet_trained_model_weights = BSON.load(target_location .* "/" .* trained_model);
Flux.loadparams!(nnet_model, nnet_trained_model_weights[:weights]);
trained_scaler = readdir(target_location);
trained_scaler = trained_scaler[contains.(trained_scaler, ".bson")][1];
trained_scaler = BSON.load(target_location .* "/" .* trained_scaler);
col_names = Symbol.(names(x));
x = MLJulFluxFun.scale_transform(trained_scaler[:x_scaler], Matrix(x));
x = DataFrame(x, col_names);
# Perform feature importance analyses in predicting MGE abundance using shapley values
shap_df = ShapML.shap(explain = x,
                        reference = x,
                        model = nnet_model,
                        predict_function = predict_function,
                        sample_size = 60,
                        seed = 1);
shap_df_mean = combine(groupby(shap_df, [:feature_name]),
:shap_effect => (x -> mean(abs.(x))) => :mean_shap_values);
sort!(shap_df_mean, [:mean_shap_values], rev = true);
shap_df_mean[!, :feature_tag] .= length(shap_df_mean.feature_name):-1:1;
shap_df = innerjoin(shap_df_mean, shap_df, on = :feature_name);
shap_df[!, :shap_flag] .= ifelse.(shap_df.shap_effect .> 0.0, 1, ifelse.(shap_df.shap_effect .< 0.0, 2, 3));
shap_df1 = combine(groupby(shap_df, [:feature_name]),
:feature_value => (x -> normalize(x)) => :feature_value_normalized,
:shap_effect => :shap_effect,
:feature_tag => :feature_tag);
show(first(shap_df1, 6), allcols = true)
feat_label_df = DataFrame(feature_name = unique(shap_df.feature_name),
feat_labels = ["Temperature (°C)", "Residence Time (days)", "Operation\n(Batch = 0, Semi-continous = 1)", "Feedstock and Co-substrate:\nFood Waste\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nFood Waste + Manure\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nFood Waste + Oil\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nFood Waste + Sewage Sludge\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nManure\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nManure + Wheat Straw\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nPharmaceutical Waste Sludge\n(No = 0, Yes = 1)", "Feedstock and Co-substrate:\nSewage Sludge\n(No = 0, Yes = 1)", "Additives/Chemicals\n(No = 0, Yes = 1)",
"Pre-treatment\n(No = 0, Yes = 1)"]);
shap_df1 = innerjoin(feat_label_df, shap_df1, on = :feature_name);
shap_df_mean = innerjoin(feat_label_df, shap_df_mean, on = :feature_name);
