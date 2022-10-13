using MattEland.ML.TimeAndSpace.Core;

namespace MattEland.ML.TimeAndSpace;

internal class AppMenu
{
    private DoctorWhoRegressionExperiment? _regressionExperiment;
    private const string DataFilePath = "WhoDataSet.csv";
    private const uint SecondsToTrain = 2;

    private string GetMainMenuChoice()
    {
        Console.WriteLine();
        Console.WriteLine("The available options are: ");
        Console.WriteLine();
        Console.WriteLine("1) Regression");
        Console.WriteLine("2) Binary Classification");
        Console.WriteLine("3) Multi-Class Classification");
        Console.WriteLine("4) Recommendation");
        Console.WriteLine("5) Ranking");
        Console.WriteLine("6) Calculate Best Episode");
        Console.WriteLine("7) Calculate Worst Episode");
        Console.WriteLine("Q) Quit");

        Console.WriteLine();
        Console.WriteLine("What do you want to do?");

        return Console.ReadLine()!;
    }

    public void RunMainMenu()
    {
        ShowWelcome();

        bool keepGoing = true;
        while (keepGoing)
        {
            string choice = GetMainMenuChoice();
            Console.WriteLine();

            switch (choice.ToUpperInvariant())
            {
                case "1": // Regression
                    PerformRegressionExperiment();
                    break;

                case "2": // Binary Classification
                    PerformBinaryClassificationExperiment();
                    break;

                case "3": // Multi-Class Classification
                    Console.WriteLine("Multi-Class Classification is not implemented");
                    break;

                case "4": // Recommendation
                    Console.WriteLine("Recommendation is not implemented");
                    break;

                case "5": // Ranking
                    Console.WriteLine("Ranking is not implemented");
                    break;

                case "6": // Calculate Best Episode
                    Console.WriteLine("Calculating BEST episode given current regression model");
                    PerformEpisodeGeneration(higherRatingIsBetter: true);
                    break;

                case "7": // Calculate Worst Episode
                    Console.WriteLine("Calculating WORST episode given current regression model");
                    PerformEpisodeGeneration(higherRatingIsBetter: false);
                    break;

                case "Q":
                    Console.WriteLine("Thanks for using the application! Allons-y!");
                    keepGoing = false;
                    break;

                default:
                    Console.WriteLine("That is not a supported input");
                    break;
            }
        }
    }

    private static void PerformBinaryClassificationExperiment()
    {
        DoctorWhoClassificationExperiment experiment = new();
        experiment.Train(DataFilePath, SecondsToTrain);

        // Predict Values from a sample episode
        Episode sampleEpisode = EpisodeBuilder.BuildSampleEpisode();

        // Get a rating prediction
        BinaryPrediction prediction = experiment.Predict(sampleEpisode);

        Console.WriteLine(prediction.Value
            ? $"This hypothetical episode WOULD take place on earth with a score of {prediction.Confidence}"
            : $"This hypothetical episode would NOT take place on earth with a score of {prediction.Confidence}");
    }

    private void PerformRegressionExperiment()
    {
        _regressionExperiment = new();
        _regressionExperiment.Train(DataFilePath, SecondsToTrain);

        // Predict Values from a sample episode
        Episode sampleEpisode = EpisodeBuilder.BuildSampleEpisode();

        // Get a rating prediction
        RegressionPrediction prediction = _regressionExperiment.Predict(sampleEpisode);
        Console.WriteLine($"This hypothetical episode would rate a {prediction.Score}");
    }

    private void PerformEpisodeGeneration(bool higherRatingIsBetter)
    {
        if (_regressionExperiment == null)
        {
            Console.WriteLine("Train a Regression Run First");
            return;
        }

        GeneticEpisodeSolver solver = new GeneticEpisodeSolver();
        Episode episode = solver.Optimize(_regressionExperiment, higherRatingIsBetter);

        Console.WriteLine("The optimized episode involves...");
        Console.WriteLine();
        Console.WriteLine(episode);
    }

    private void ShowWelcome()
    {
        // Welcome text
        Console.WriteLine("Hello, Time and Space!");
        Console.WriteLine();
        Console.WriteLine("This app is intended to demonstrate ML.NET AutoML and is not intended for serious usage");
        Console.WriteLine("Doctor Who and associated properties and trademarks are Copyright of the British Broadcasting Corporation (BBC)");
        Console.WriteLine();
    }
}