using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MattEland.ML.Common
{
    public class BinaryClassificationConsoleProgressHandler : IProgress<RunDetail<BinaryClassificationMetrics>>
    {
        public void Report(RunDetail<BinaryClassificationMetrics> value)
        {
            // When the analysis completes, the ValidationMetrics property will be null
            if (value.ValidationMetrics == null)
            {
                Console.WriteLine("Finished evaluating");
                return;
            }

            Console.WriteLine($"{value.TrainerName} ran in {value.RuntimeInSeconds} seconds");
            //value.ValidationMetrics.LogMetricsString(prefix: "\t");
        }
    }
}
