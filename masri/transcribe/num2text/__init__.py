# MASRI/masri/transcribe/num2text/__init__.py (https://github.com/UMSpeech/MASRI/blob/2e6cf2afe72c3c9d222f002018f8febbc9c48a61/masri/transcribe/num2text/__init__.py)

from . import lang_MT

def num2text(number, lang='mt', to='cardinal', **kwargs):
    converter = lang_MT.Num2Word_MT()
    number = str(number)
    number = number.replace(",", "")
    number = converter.str_to_number(number)

    return getattr(converter, 'to_{}'.format(to))(number, **kwargs)