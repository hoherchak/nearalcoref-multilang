import stanza

for lang in ['en', 'uk', 'ru']:
    stanza.download(
        lang=lang,
        dir='../models'
    )
