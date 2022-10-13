using MattEland.ML.Common;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MattEland.ML.TimeAndSpace.Core;

public class DoctorWhoRegressionExperiment : DoctorWhoExperimentBase
{
    private PredictionEngine<Episode, RegressionPrediction>? _predictionEngine;

    public void Train(string dataPath, uint secondsToTrain=30, bool showDetailedProgress = false)
    {
        // Load our source data and split it for training
        DataOperationsCatalog.TrainTestData trainTest = LoadTrainTestData(dataPath);

        // Regression - Predict the Rating of a Doctor Who episode
        // Configure experiment
        RegressionExperimentSettings settings = new()
        {
            MaxExperimentTimeInSeconds = secondsToTrain,
            OptimizingMetric = RegressionMetric.RSquared,
        };

        RegressionExperiment experiment = Context.Auto().CreateRegressionExperiment(settings);

        // Train a model
        Console.WriteLine($"Training for {secondsToTrain} seconds...");
        Console.WriteLine();

        ExperimentResult<RegressionMetrics> result = experiment.Execute(
            trainData: trainTest.TrainSet,
            validationData: trainTest.TestSet,
            labelColumnName: nameof(Episode.Rating),
            preFeaturizer: null,
            progressHandler: new RegressionConsoleProgressHandler(showDetailedProgress: showDetailedProgress));

        Console.WriteLine();
        Console.WriteLine("Finished Training!");

        // Evaluate Results
        Console.WriteLine();
        Console.WriteLine($"Best algorithm: {result.BestRun.TrainerName}{Environment.NewLine}");
        result.BestRun.ValidationMetrics.LogMetricsString();

        /* PFI Code does not currently work with AutoML
        IDataView fullData = Context.Data.LoadCsv<Episode>(dataPath);
        ImmutableDictionary<string, RegressionMetricsStatistics>? pfi =
            Context
                .Regression
                .PermutationFeatureImportance(model: result.BestRun.Model, data: fullData, permutationCount: 3, labelColumnName: nameof(Episode.Rating));

        foreach (KeyValuePair<string, RegressionMetricsStatistics> kvp in pfi)
        {
            Console.WriteLine(kvp.Key + ": " + kvp.Value.ToString());
        }
        */

        // Build a Prediction Engine to predict new values
        _predictionEngine =
            Context.Model.CreatePredictionEngine<Episode, RegressionPrediction>(
                transformer: result.BestRun.Model,
                inputSchema: trainTest.TestSet.Schema
            );
    }

    public RegressionPrediction Predict(Episode sampleEpisode)
    {
        if (_predictionEngine == null)
            throw new InvalidOperationException("Cannot make predictions when the model hasn't been trained");

        return _predictionEngine.Predict(sampleEpisode);
    }
}