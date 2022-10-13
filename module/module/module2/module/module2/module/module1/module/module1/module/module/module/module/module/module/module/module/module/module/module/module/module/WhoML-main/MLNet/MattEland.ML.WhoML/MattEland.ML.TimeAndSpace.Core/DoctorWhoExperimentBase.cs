using MattEland.ML.Common;
using Microsoft.ML;

namespace MattEland.ML.TimeAndSpace.Core;

public abstract class DoctorWhoExperimentBase
{
    protected MLContext Context { get; } = new();

    protected DataOperationsCatalog.TrainTestData LoadTrainTestData(string path)
    {
        IDataView rawData = Context.Data.LoadCsv<Episode>(path: path);
        DataOperationsCatalog.TrainTestData trainTest = Context.Data.TrainTestSplit(
            data: rawData,
            testFraction: 0.2,
            samplingKeyColumnName: null);

        return trainTest;
    }

}