using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MattEland.ML.Common
{
    public class RegressionConsoleProgressHandler : IProgress<RunDetail<RegressionMetrics>>
    {
        private readonly bool _showDetailedProgress;

        public RegressionConsoleProgressHandler(bool showDetailedProgress = false)
        {
            _showDetailedProgress = showDetailedProgress;
        }

        public void Report(RunDetail<RegressionMetrics> value)
        {
            // When the analysis completes, the ValidationMetrics property will be null
            if (value.ValidationMetrics == null) return;

            Console.WriteLine($"{value.TrainerName} ran in {value.RuntimeInSeconds} seconds");

            if (_showDetailedProgress)
            {
                value.ValidationMetrics.LogMetricsString(prefix: "\t");
            }
        }
    }
}
