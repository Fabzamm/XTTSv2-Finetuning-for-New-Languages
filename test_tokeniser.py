from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

tokenizer = VoiceBpeTokenizer()

sentences = [
    "Għal sitt snin sħaħ iz-zija Anna għexet waħidha.",
    "Xemx u xita Alla jaf meta.",
    "Dawret iċ-ċavetta tal-bieba u daħlet.",
    "Kif marret il-mużika Maltija f'dawn l-aħħar ħmistax-il sena?",
    "Għalkemm il-mowbajl jista' jkun ta' għajnuna f'każijiet urġenti, sfortunatament sar ukoll il-kawża ta' inċidenti.",
    "Għar-raba' sena konsekuttiva kien hemm tnaqqis sinifikanti fir-rapporti ta' reati kriminali.",
    "L-anzjani ta' San Vincenz bdew jieħdu t-tieni doża tat-tilqima.",
    "Toni tagħna tani tina talli tajtu tuta tajba.",
    "Hemm madwar erbgħin miljun għasfur domestiku.",
    "L-ispettur fetaħ in-nowtbuk u beda jieħu n-noti.",
    "Sibt xogħol ta' skrivan ma' nutar xiħ.",
    "Il-karozzi tal-linja mhumiex il-problema, fil-fatt jipprovdu soluzzjoni.",
    "Kielu r-ross il-forn fis-skiet.",
    "Għamel biċċa ħobż bil-ġobon u l-perżut.",
    "Il-qerda tal-foresti iġġib tibdil fil-klima lokali u fiċ-ċiklu tal-ilma.",
    "Wieġeb kull mistoqsija fi kliemek.",
    "Fetaħ pakkett biskuttini u beda jbillhom fit-te.",
    "Fl-irħula antiki wieħed jista' jsib bosta toroq dojoq.",
    "Illum il-ġurnata l-kompjuter sar għodda li ma nistgħux ngħaddu mingħajrha.",
    "Ħafna persuni qed jinvestu f'enerġija alternattiva.",

    "Din il-ġrajja kollha jiena sirt naf biha meta kelli xi erbatax-il sena.",
    "Kelli nofstanhar mimli laqgħat u appuntamenti.",
    "Dik il-ħabta l-uġigħ ta' rasijiet kienu fl-aqwa tagħhom.",
    "Wara t-tieni Gwerra Dinjija kien hawn ħafna faqar.",
    "Jeħtieġ li nagħmlu evalwazzjoni profonda tal-programmi ta' riabilitazzjoni.",
    "Fix-xhur tas-sajf hemm tendenza li l-konsum tal-alkoħol ikun ogħla mill-bqija tas-sena.",
    "L-influwenza internazzjonali se titnaqqas.",
    "Kienu tlieta l-istejjer li ppreżentaw il-ġurnalisti.",
    "L-ewwel pesta kbira tas-seklu għoxrin kienet l-influwenza Spanjola.",
    "Bħall-bnedmin, l-annimali jafu u japprezzaw lil min iħobbhom.",
    "Il-problema tal-għargħar ħakmet ukoll lill-Brażil.",
    "Kull student għandu jħossu komdu jaqra silta minn ktieb lil sħabu tal-klassi.",
    "Il-ktieb nistgħu ngħidu li huwa l-aqwa invenzjoni tal-bniedem.",
    "Il-Kunsill Nazzjonali tal-Ktieb jinkoraġġixxi awturi ġodda biex iwasslu x-xogħlijiet tagħhom għall-attenzjoni tal-pubbliku.",
    "Il-bilingwiżmu, b'mod speċjali fi żmienna, żmien ta' globalizzazzjoni, qiegħed kulma jmur isir dejjem iktar importanti.",
    "Huwa importanti li ngħidu li l-ilsien Malti u l-ilsien Ingliż huma ż-żewġ lingwi uffiċjali ta' Malta.",
    "L-element Semitiku huwa l-qofol tal-ilsien Malti.",
    "Waħda mill-aktar problemi serji li qiegħda taffaċċja d-dinja llum fl-ambjent hija t-tibdil tal-klima.",
    "L-arloġġ tal-kampnar għadu kemm kien restawrat.",
    "Għadha kemm ġiet iċċelebrata l-festa tal-patrun tar-raħal.",
    "Ma kienx fadallu aktar ġebel xi jgara.",
    "Il-patrijiet u s-sorijiet missjunarji kienu jgħallmu lit-tfal.",
    "Fl-aħħar snin pajjiżna għamel passi 'l quddiem f'ċerti oqsma.",
    "F'ħajtu kulħadd ikollu mumenti sbieħ u oħrajn diffiċli.",
    "Waslet lura d-dar meta kien ġa daqq nofsinhar.",
    "Żagħżugħ jindarab gravi wara li waqa' minn fuq sellum.",
    "Fuq il-post issejjaħ tim mediku li taw l-ewwel għajnuna u ambulanza li ħadet lill-vittma l-Isptar.",
    "L-awtoritajiet tas-saħħa ħabbru li mitejn u ħdax-il persuna oħra ġew iddikjarati mfejqa.",
    "Dawn in-novelli kienu ppublikati għall-ewwel darba.",
    "Xtara maħżen kbir fil-kampanja.",
]

# Process + lowercase
processed = [tokenizer.preprocess_text(s, "mt").lower() for s in sentences]

# Print in your desired format
for p in processed:
    print(f'"{p}",')
