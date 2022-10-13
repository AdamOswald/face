def BFEx2SansLang(self, code):
    BFEXcmd = list("><v^+-*[].,=")
    commands = "샌즈.알피스.파피루스.메타톤.와.토비폭스.겁나.차라.프리스크.언더테일.모르시는구나.어렵습니다".split(".")
    cmddict = {BFEXcmd[i]:commands[i] for i in range(len(commands))}
    now, cnt, res = "", 0, []
    for i in code + "?":
        if i == now: cnt += 1
        else: res.append([now,cnt]); now = i; cnt = 1
    return ' '.join(list(map(lambda x:cmddict[x[0]] + "!"*(x[1]-1),res[1:]))).strip() + " 아시는구나! 참고로 겁나 어렵습니다"
