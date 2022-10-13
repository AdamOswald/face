using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MattEland.ML.TimeAndSpace;

public static class InputHelper
{
    public static uint GetUnsignedInteger(string prompt, uint minValue = 0)
    {
        int value;
        string? userInput = null;
        do
        {
            if (userInput != null)
            {
                Console.WriteLine($"Please enter a positive number greater than {minValue}");
            }

            Console.WriteLine(prompt);
            userInput = Console.ReadLine();
        } while (!int.TryParse(userInput, out value) || value < minValue);

        return (uint) value;
    }
}