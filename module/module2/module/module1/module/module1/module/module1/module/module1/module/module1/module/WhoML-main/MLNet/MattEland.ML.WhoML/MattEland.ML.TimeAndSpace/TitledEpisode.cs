using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MattEland.ML.TimeAndSpace.Core;

namespace MattEland.ML.TimeAndSpace;

public class TitledEpisode : Episode
{
    public string Title { get; set; }

    public string Doctors 
    { 
        get
        {
            StringBuilder sb = new();

            if (Has9) sb.Append("9, ");
            if (Has10) sb.Append("10, ");
            if (Has11) sb.Append("11, ");
            if (Has12) sb.Append("12, ");
            if (Has13) sb.Append("13, ");
            if (HasWarDoctor) sb.Append("the War Doctor");

            return sb.ToString();
        }
    }

    public string Companions 
    { 
        get
        {
            StringBuilder sb = new();

            if (HasAmy) sb.Append("Amy, ");
            if (HasBill) sb.Append("Bill, ");
            if (HasChurchill) sb.Append("Churchill, ");
            if (HasClara) sb.Append("Clara, ");
            if (HasDanny) sb.Append("Danny, ");
            if (HasDonna) sb.Append("Donna, ");
            if (HasGrace) sb.Append("Grace, ");
            if (HasGraham) sb.Append("Graham, ");
            if (HasJackie) sb.Append("Jackie, ");
            if (HasJenny) sb.Append("Jenny, ");
            if (HasKate) sb.Append("Kate, ");
            if (HasMartha) sb.Append("Martha, ");
            if (HasMickey) sb.Append("Mickey, ");
            if (HasMadameVastra) sb.Append("Madame Vastra, ");
            if (HasNardole) sb.Append("Nardole, ");
            if (HasOod) sb.Append("Ood, ");
            if (HasOsgood) sb.Append("Osgood, ");
            if (HasRiver) sb.Append("River, ");
            if (HasRory) sb.Append("Rory, ");
            if (HasRose) sb.Append("Rose, ");
            if (HasRyan) sb.Append("Ryan, ");
            if (HasSarah) sb.Append("Sarah, ");
            if (HasChurchill) sb.Append("Sophie, ");
            if (HasYasmine) sb.Append("Yasmine, ");

            return sb.ToString();
        }
    }
    public string Opponents 
    { 
        get
        {
            StringBuilder sb = new();

            if (HasDalek) sb.Append("Daleks, ");
            if (HasCybermen) sb.Append("Cybermen, ");
            if (HasZygon) sb.Append("Zygons, ");
            if (HasTheMaster) sb.Append("The Master, ");
            if (HasTheSilent) sb.Append("The Silent, ");
            if (HasWeepingAngels) sb.Append("Weeping Angels, ");
            if (HasMadameKovorian) sb.Append("Madame Kovorian, ");
            if (HasJudoon) sb.Append("Judoon, ");
            if (HasSontaran) sb.Append("Sontarans, ");
            
            return sb.ToString();
        }
    }
}