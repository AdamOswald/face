using Microsoft.ML.Data;

namespace MattEland.ML.TimeAndSpace.Core;

public class BinaryPrediction
{
    [ColumnName("Score")]
    public float Confidence { get; set; }

    [ColumnName("PredictedLabel")]
    public bool Value { get; set; }
}