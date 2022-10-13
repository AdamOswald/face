using System.Diagnostics;
using MattEland.ML.Common;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MattEland.ML.TimeAndSpace.Core;

public class DoctorWhoClassificationExperiment : DoctorWhoExperimentBase
{
    private PredictionEngine<Episode, BinaryPrediction>? _predictionEngine;

    public void Train(string dataPath, uint secondsToTrain = 30)
    {
        // Load our source data and split it for training
        DataOperationsCatalog.TrainTestData trainTest = LoadTrainTestData(dataPath);

        // Binary Classification - Predict if an episode takes place in the present
        // Configure experiment
        BinaryExperimentSettings settings = new()
        {
            MaxExperimentTimeInSeconds = secondsToTrain,
            OptimizingMetric = BinaryClassificationMetric.F1Score,
        };

        BinaryClassificationExperiment experiment = Context.Auto().CreateBinaryClassificationExperiment(settings);

        // Train a model
        Console.WriteLine($"Training for {secondsToTrain} seconds...");

        ExperimentResult<BinaryClassificationMetrics> result = experiment.Execute(
            trainData: trainTest.TrainSet,
            validationData: trainTest.TestSet,
            labelColumnName: nameof(Episode.IsPresent),
            preFeaturizer: null,
            progressHandler: new BinaryClassificationConsoleProgressHandler());

        // Evaluate Results
        Console.WriteLine($"Best algorithm: {result.BestRun.TrainerName}{Environment.NewLine}");
        //result.BestRun.ValidationMetrics.LogMetricsString();

        // Build a Prediction Engine to predict new values
        _predictionEngine =
            Context.Model.CreatePredictionEngine<Episode, BinaryPrediction>(
                transformer: result.BestRun.Model,
                inputSchema: trainTest.TestSet.Schema
            );
    }

    public BinaryPrediction Predict(Episode sampleEpisode)
    {
        if (_predictionEngine == null)
            throw new InvalidOperationException("Cannot make predictions when the model hasn't been trained");

        return _predictionEngine.Predict(sampleEpisode);
    }
}