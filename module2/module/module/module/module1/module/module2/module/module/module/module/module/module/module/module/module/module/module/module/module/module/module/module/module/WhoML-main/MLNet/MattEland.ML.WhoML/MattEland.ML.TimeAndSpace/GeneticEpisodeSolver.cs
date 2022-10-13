using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GeneticSharp.Domain;
using GeneticSharp.Domain.Chromosomes;
using GeneticSharp.Domain.Crossovers;
using GeneticSharp.Domain.Fitnesses;
using GeneticSharp.Domain.Mutations;
using GeneticSharp.Domain.Populations;
using GeneticSharp.Domain.Randomizations;
using GeneticSharp.Domain.Selections;
using GeneticSharp.Domain.Terminations;
using MattEland.ML.TimeAndSpace.Core;

namespace MattEland.ML.TimeAndSpace
{
    public class EpisodeScoreFitness : IFitness
    {
        private readonly DoctorWhoRegressionExperiment _experiment;
        private readonly bool _higherRatingIsBetter;

        public EpisodeScoreFitness(DoctorWhoRegressionExperiment experiment, bool higherRatingIsBetter)
        {
            _experiment = experiment;
            _higherRatingIsBetter = higherRatingIsBetter;
        }

        public double Evaluate(IChromosome chromosome)
        {
            EpisodeTraitsChromosome epChromisome = (EpisodeTraitsChromosome) chromosome;
            var episode = epChromisome.BuildEpisode();

            float score = _experiment.Predict(episode).Score;

            if (!_higherRatingIsBetter)
            {
                score *= -1;
            }

            return score;
        }
    }

    public class EpisodeTraitsChromosome : ChromosomeBase
    {
        public EpisodeTraitsChromosome() : base(length: 4)
        {
        }

        public override Gene GenerateGene(int geneIndex)
        {
            return new Gene(RandomizationProvider.Current.GetInt(0, 2));
        }

        public override IChromosome CreateNew()
        {
            return new EpisodeTraitsChromosome();
        }

        public Episode BuildEpisode()
        {
            return new Episode
            {
                IsSpace = GetGene(0).Value == (object) 1,
                IsPast = GetGene(1).Value == (object)1,
                IsPresent = GetGene(1).Value == (object)1,
                IsFuture = GetGene(1).Value == (object)1,
            };
        }
    }

    public class GeneticEpisodeSolver
    {
        public Episode Optimize(DoctorWhoRegressionExperiment experiment, bool higherRatingIsBetter)
        {
            var population = new Population(50, 100, adamChromosome: new EpisodeTraitsChromosome());
            var crossover = new UniformCrossover(0.5f);
            var selection = new EliteSelection();
            var mutation = new FlipBitMutation();
            var fitness = new EpisodeScoreFitness(experiment, higherRatingIsBetter);

            var ga = new GeneticAlgorithm(population: population,
                                          fitness: fitness,
                                          selection: selection,
                                          crossover: crossover,
                                          mutation: mutation);

            var termination = new FitnessStagnationTermination(25);
            ga.Termination = termination;

            ga.GenerationRan += (sender, args) =>
            {
                Console.WriteLine("Finished a generation at " + DateTime.Now.ToLongTimeString());
            };

            ga.TerminationReached += (sender, args) =>
            {
                Console.WriteLine("Termination Reached");
            };

            ga.Start();

            while (ga.IsRunning)
            {

            }

            EpisodeTraitsChromosome best = (EpisodeTraitsChromosome) ga.BestChromosome;

            return best.BuildEpisode();
        }
    }
}
