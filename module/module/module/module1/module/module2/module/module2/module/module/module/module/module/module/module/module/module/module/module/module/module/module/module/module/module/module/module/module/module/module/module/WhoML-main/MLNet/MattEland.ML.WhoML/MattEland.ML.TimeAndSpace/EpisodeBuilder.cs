using Bogus;
using MattEland.ML.TimeAndSpace.Core;

namespace MattEland.ML.TimeAndSpace;

public static class EpisodeBuilder
{
    private static Faker<TitledEpisode>? _builder;

    public static TitledEpisode BuildRandomEpisode()
    {
        Faker<TitledEpisode> builder = _builder ?? InitializeBuilder();

        return builder.Generate();
    }

    private static Faker<TitledEpisode> InitializeBuilder()
    {
        string[] musicians = {"Murray Gold", "Segun Akinola"};

        string[] producers = {"Phil Collinson", "Susie Liggat", "Tracie Simpson",
            "Peter Bennett", "Patrick Schweitzer", "Sanne Wohlenberg",
            "Marcus Wilson", "Denise Paul", "Nikki Wilson", "Paul Frift",
            "Derek Ritchie", "Nicola Wilson", "Alex Mercer", "Pete Levy",
            "Sheena Bucktowonsing"};        

        string[] writers = {"Russell T Davies", "Mark Gatiss", "Robert Shearman",
            "Paul Cornell", "Steven Moffat", "Toby Whithouse", "Tom MacRae",
            "Matt Jones", "Matthew Graham", "Gareth Roberts", "Helen Raynor",
            "Stephen Greenhorn", "Chris Chibnall", "James Moran",
            "Keith Temple", "Simon Nye", "Richard Curtis", "Steve Thompson",
            "Neil Gaiman", "Neil Cross", "Peter Harness", "Jamie Mathieson",
            "Frank Cottrell-Boyce", "Catherine Tregenna", "Sarah Dollard",
            "Mike Bartlett", "Rona Munro", "Vinay Patel", "Pete McTighe",
            "Joy Wilkinson", "Ed Hime", "Nina Metivier", "Charlene James",
            "Maxine Alderton"};

        string[] directors = {
            "Keith Boak", "Euros Lyn", "Joe Ahearne", "Brian Grant",
            "James Hawes", "Graeme Harper", "James Strong", "Dan Zeff",
            "Charles Palmer", "Richard Clark", "Hettie MacDonald",
            "Colin Teague", "Douglas Mackinnon", "Alice Troughton",
            "Adam Smith", "Andrew Gunn", "Jonny Campbell",
            "Catherine Morshead", "Ashley Way", "Toby Haynes", "Jeremy Webb",
            "Julian Simpson", "Peter Hoar", "Richard Senior", "Nick Hurran",
            "Steve Hughes", "Farren Blackburn", "Saul Metzstein",
            "Colm McCarthy", "Jamie Payne", "Mat King", "Stephen Woolfenden",
            "Ben Wheatley", "Paul Murphy", "Paul Wilmshurst", "Sheree Folkson",
            "Rachel Talalay", "Daniel O'Hara", "Ed Bazalgette",
            "Daniel Nettheim", "Justin Molotnikov", "Lawrence Gough",
            "Bill Anderson", "Wayne Yip", "Jamie Childs", "Mark Tonderai",
            "Sallie Aprahamian", "Jennifer Perrott", "Wayne Che Yip",
            "Jamie Magnus Stone", "Lee Haven Jones", "Nida Manzoor",
            "Emma Sullivan", "Azhur Saleem", "Annetta Laufer"
        };

        _builder = new Faker<TitledEpisode>()
            .RuleForType(typeof(bool), f => f.Random.Bool())
            .RuleFor(e => e.DayAired, f => f.Date.Weekday())
            .RuleFor(e => e.Title, f => f.Commerce.ProductName())
            .RuleFor(e => e.Music, f => f.Random.ArrayElement(musicians))
            .RuleFor(e => e.Writer, f => f.Random.ArrayElement(writers))
            .RuleFor(e => e.Producer, f => f.Random.ArrayElement(producers))
            .RuleFor(e => e.Director, f => f.Random.ArrayElement(directors));

        return _builder;
    }

    public static Episode BuildSampleEpisode() =>
        new()
        {
            DayAired = "Friday",
            Has11 = true,
            HasTheMaster = true,
            HasAmy = true,
            HasRory = true,
            HasRiver = true,
            HasWarDoctor = true,
            HasGraham = true,
            HasSontaran = true,
            Producer = "Alex Mercer",
            Writer = "Stephen Moffat",
            Music = "Murray Gold",
            IsFuture = true,
            IsSpace = true,
        };
}