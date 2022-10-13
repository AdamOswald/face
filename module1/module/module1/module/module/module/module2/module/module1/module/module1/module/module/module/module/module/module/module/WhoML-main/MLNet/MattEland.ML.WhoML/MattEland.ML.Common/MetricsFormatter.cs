using System.Text;
using Microsoft.ML.Data;

namespace MattEland.ML.Common
{
    public static class MetricsFormatter
    {
        public static string BuildMetricsString(this RegressionMetrics metrics, string prefix = "")
        {
            StringBuilder sb = new();
            sb.AppendLine($"{prefix}R Squared (Coefficient of Determination): {metrics.RSquared}");
            sb.AppendLine($"{prefix}Mean Absolute Error (MAE): {metrics.MeanAbsoluteError}");
            sb.AppendLine($"{prefix}Mean Squared Error (MSE): {metrics.MeanSquaredError}");
            sb.AppendLine($"{prefix}Root Mean Squared Error (RMSE): {metrics.RootMeanSquaredError}");
            sb.AppendLine($"{prefix}Loss Function: {metrics.LossFunction}");

            return sb.ToString();
        }

        public static void LogMetricsString(this RegressionMetrics metrics, string prefix = "")
        {
            Console.WriteLine(metrics.BuildMetricsString(prefix: prefix));
        }
    }
}