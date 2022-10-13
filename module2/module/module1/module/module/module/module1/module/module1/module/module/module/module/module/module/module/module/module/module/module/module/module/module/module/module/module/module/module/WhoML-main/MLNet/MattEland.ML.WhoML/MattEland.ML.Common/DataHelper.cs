using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;

namespace MattEland.ML.Common
{
    public static class DataHelper
    {
        public static IDataView LoadCsv<T>(this DataOperationsCatalog data,
            string path,
            bool hasHeader = true,
            bool allowQuoting = true,
            bool trimWhitespace = true)
        {
            return data.LoadFromTextFile<T>(path: path,
                separatorChar: ',',
                hasHeader: hasHeader,
                allowQuoting: allowQuoting,
                trimWhitespace: trimWhitespace);
        }

    }
}