using MattEland.ML.TimeAndSpace;
using MattEland.ML.TimeAndSpace.Core;
using Newtonsoft.Json;

// Train a regression model to predict episode scores based on historic episodes
uint secondsToTrain = InputHelper.GetUnsignedInteger("How many seconds do you want to train? (10 - 30 recommended)", minValue: 1);
DoctorWhoRegressionExperiment experiment = new();
experiment.Train(dataPath: "WhoDataSet.csv", secondsToTrain: secondsToTrain);

// Generate a bunch of episodes, keeping only a season's worth of the best options
uint episodesToKeep = InputHelper.GetUnsignedInteger("How many episodes are in your season? (At least 1)", minValue: 1);
uint episodesToGenerate = InputHelper.GetUnsignedInteger($"How many episodes do you want to generate? (At least {episodesToKeep})", minValue: episodesToKeep);
Console.WriteLine($"Generating a season of {episodesToKeep} episodes from {episodesToGenerate} candidates");

List<TitledEpisode> bestEpisodes = new();
List<TitledEpisode> worstEpisodes = new();
for (int i = 0; i < episodesToGenerate; i++)
{
    // Build a random episode
    TitledEpisode ep = EpisodeBuilder.BuildRandomEpisode();

    // Generate a predicted score for that episode
    RegressionPrediction prediction = experiment.Predict(ep);
    ep.Rating = prediction.Score;

    // If this is one of the best episodes that we have, or the list isn't full yet, add it to the season
    if (bestEpisodes.Count < episodesToKeep)
    {
        bestEpisodes.Add(ep);
    } 
    else if (ep.Rating > bestEpisodes.Min(e => e.Rating))
    {
        bestEpisodes.Add(ep);
        bestEpisodes = bestEpisodes.OrderByDescending(e => e.Rating).Take((int)episodesToKeep).ToList();
    }

    // If this is one of the worst episodes that we have, or the list isn't full yet, add it to the season
    if (worstEpisodes.Count < episodesToKeep)
    {
        worstEpisodes.Add(ep);
    } 
    else if (ep.Rating < worstEpisodes.Max(e => e.Rating))
    {
        worstEpisodes.Add(ep);
        worstEpisodes = worstEpisodes.OrderBy(e => e.Rating).Take((int)episodesToKeep).ToList();
    }
}

// Display the best episodes
Console.WriteLine("The best episode involves...");

TitledEpisode bestEpisode = bestEpisodes.OrderByDescending(e => e.Rating).First();
Console.WriteLine(JsonConvert.SerializeObject(bestEpisode, Formatting.Indented));

// Serialize the full season to disk
File.WriteAllText("best_season.json", JsonConvert.SerializeObject(bestEpisodes, Formatting.Indented));
File.WriteAllText("worst_season.json", JsonConvert.SerializeObject(worstEpisodes, Formatting.Indented));
Console.WriteLine();
Console.WriteLine("Saved full season to season.json for inspection");

// Closing
Console.WriteLine();
Console.WriteLine($"Have fun watching {bestEpisode.Title}! Allons-y!" );