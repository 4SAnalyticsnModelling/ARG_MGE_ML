# This script trains RF (Random Forest), XGBOOST(Extreme Gradient Boost), and NNET(Artificial Neural Network) regression models to predict abundance of MGE
# Install required packages
using Pkg;
Pkg.add(["DataFrames", "CSV", "Statistics", "BSON", "MLJ", "MLJXGBoostInterface", "MLJDecisionTreeInterface", "Flux", "Flux.Zygote"]);
Pkg.add("https://github.com/4SAnalyticsnModelling/MLJulFluxFun"); #This is a module which was developed as a part of this project. All the functions used below for model training are imported from the MLJulFluxFun module.
# Import the required packages
using DataFrames, CSV, Statistics, BSON, MLJ, MLJXGBoostInterface, MLJDecisionTreeInterface, Flux, Flux.Zygote, MLJulFluxFun;
# Set current directory as working directory
cd(@__DIR__);
# Import the dataset for model training
arg_mge_df = DataFrame(CSV.File("../../assets/arg_mge_data_for_ML.csv", normalizenames = true));
# Process the dataset for training
mge_df0 = arg_mge_df[!, [:temperature, :residence_time, :operation, :feedstock_and_co_substrate, :additives_or_chemicals, :pre_treatment, :mge_removal_efficiency]];
mge_df0 = mge_df0[mge_df0[!, :mge_removal_efficiency] .< 4000, :];
mge_df0[!, :operation] .= ifelse.(mge_df0.operation .== "Batch", 0.0, 1.0);
mge_df0[!, :additives_or_chemicals] .= ifelse.(mge_df0.additives_or_chemicals .== "No", 0.0, 1.0);
mge_df0[!, :pre_treatment] .= ifelse.(mge_df0.pre_treatment .== "No", 0.0, 1.0);
list_cols = [:feedstock_and_co_substrate]
for items in list_cols
    mge_df0[!, items] .= MLJ.categorical(mge_df0[!, items])
end
# One hot encoding
hot = MLJ.OneHotEncoder(ordered_factor = true);
mach = MLJ.fit!(machine(hot, mge_df0), verbosity = 0);
mge_df = MLJ.transform(mach, mge_df0);
show(first(mge_df, 6), allcols = true)

# ML training for predicting abundance in MGE
x = mge_df[:, 1:13];
y = mge_df[:, 14];
param_of_interest = "MGE";

# Random Forest (RF) model training and cross-validation
model_name = "random_forest";
# Build the model
forest_model = MLJDecisionTreeInterface.RandomForestRegressor();
# 20% holdout cross-validation
mod_perform_df = MLJulFluxFun.mlj_mod_eval(forest_model, x, y, param_of_interest * "_" * model_name * "_training_performance_holdout", forest_model.n_trees, :ntree, 705:1:705, MLJulFluxFun.Holdout(eachindex(y), 0.8, true, 500));
# 5-fold cross-validation
mod_perform_df = MLJulFluxFun.mlj_mod_eval(forest_model, x, y, param_of_interest * "_" * model_name * "_training_performance_kfold", forest_model.n_trees, :ntree, 705:1:705, MLJulFluxFun.KFold(eachindex(y), 5, true, 100));
# Final trained model with tuned hyperparameters
mod_perform_df = MLJulFluxFun.mlj_mod_eval(forest_model, x, y, param_of_interest * "_" * model_name * "_trained_model", forest_model.n_trees, :ntree, 705:1:705);

# Extreme Gradient Boost (XGBOOST) model training and cross-validation
model_name = "xgboost";
# Build the model
xgboost_model = MLJXGBoostInterface.XGBoostRegressor();
# 20% holdout cross-validation
mod_perform_df = MLJulFluxFun.mlj_mod_eval(xgboost_model, x, y, param_of_interest * "_" * model_name * "_training_performance_holdout", xgboost_model.eta, :etas, 0.1:0.1:0.1, MLJulFluxFun.Holdout(eachindex(y), 0.8, true, 500));
# 5-fold cross-validation
mod_perform_df = MLJulFluxFun.mlj_mod_eval(xgboost_model, x, y, param_of_interest * "_" * model_name * "_training_performance_kfold", xgboost_model.eta, :etas, 0.1:0.1:0.1, MLJulFluxFun.KFold(eachindex(y), 5, true, 100));
# Final trained model with tuned hyperparameters
mod_perform_df = MLJulFluxFun.mlj_mod_eval(xgboost_model, x, y, param_of_interest * "_" * model_name * "_trained_model", xgboost_model.eta, :etas, 0.1:0.1:0.1);

# Artficial Neural Network (NNET) model training and cross-validation
model_name = "nnet"
# Build the model
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
# 20% holdout cross-validation
MLJulFluxFun.flux_mod_eval(nnet_build(nnet_mod_builder(N * 6, N * 4, N * 2), N, 1), x, y, param_of_interest * "_" * model_name * "_training_performance_holdout", MLJulFluxFun.Holdout(eachindex(y), 0.8, true, 500), 100, true, MLJulFluxFun.standard_scaler(), nothing, 5, trunc(Int64, round(J * 0.8)), 3,
2, Flux.Losses.mse, Flux.Optimise.ADAM(0.001));
# 5-fold cross-validation
MLJulFluxFun.flux_mod_eval(nnet_build(nnet_mod_builder(N * 6, N * 4, N * 2), N, 1), x, y, param_of_interest * "_" * model_name * "_training_performance_kfold", MLJulFluxFun.KFold(eachindex(y), 5, true, 100), 100, true, MLJulFluxFun.standard_scaler(), nothing, 5, div(J, 5) * 4, 3,
2, Flux.Losses.mse, Flux.Optimise.ADAM(0.001));
# Final trained model with tuned hyperparameters
MLJulFluxFun.flux_mod_eval(nnet_build(nnet_mod_builder(N * 6, N * 4, N * 2), N, 1), x, y, param_of_interest * "_" * model_name * "_trained_model", nothing, 650, false, MLJulFluxFun.standard_scaler(), nothing, 10, length(y), 3, 2, Flux.Losses.mse,
Flux.Optimise.ADAM(0.001));
