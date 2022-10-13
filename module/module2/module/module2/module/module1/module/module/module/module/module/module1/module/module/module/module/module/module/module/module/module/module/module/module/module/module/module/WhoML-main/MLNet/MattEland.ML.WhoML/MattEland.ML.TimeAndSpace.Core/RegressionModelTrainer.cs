using System.Diagnostics.CodeAnalysis;
using MattEland.ML.Common;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MattEland.ML.TimeAndSpace.Core;

public static class RegressionModelTrainer
{
    [SuppressMessage("ReSharper.DPA", "DPA0002: Excessive memory allocations in SOH", MessageId = "type: Microsoft.ML.AutoML.StringParameterValue; size: 158MB")]
    public static PredictionEngine<Episode, RegressionPrediction> TrainDoctorWhoRegressionPredictor(
        this MLContext context,
        DataOperationsCatalog.TrainTestData trainTest)
    {
        // Configure experiment
        const uint secondsToTrain = 10;
        RegressionExperimentSettings settings = new()
        {
            MaxExperimentTimeInSeconds = secondsToTrain,
            OptimizingMetric = RegressionMetric.RSquared,
        };

        RegressionExperiment experiment = context.Auto().CreateRegressionExperiment(settings);

        // Train a model
        Console.WriteLine($"Training for {secondsToTrain} seconds...");

        ExperimentResult<RegressionMetrics> result = experiment.Execute(
            trainData: trainTest.TrainSet,
            validationData: trainTest.TestSet,
            labelColumnName: nameof(Episode.Rating),
            preFeaturizer: null,
            progressHandler: new RegressionConsoleProgressHandler());

        // Evaluate Results
        Console.WriteLine($"Best algorithm: {result.BestRun.TrainerName}{Environment.NewLine}");
        result.BestRun.ValidationMetrics.LogMetricsString();

        // Build a Prediction Engine to predict new values
        PredictionEngine<Episode, RegressionPrediction> predictionEngine =
            context.Model.CreatePredictionEngine<Episode, RegressionPrediction>(
                transformer: result.BestRun.Model,
                inputSchema: trainTest.TestSet.Schema
            );

        return predictionEngine;
    }
}