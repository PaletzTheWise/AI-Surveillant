#!/bin/bash

if [ ! -d locales/$lang ]; then
    mkdir locales/$lang
fi
echo "" > locales/AISurveillant.pot
xgettext --keyword=get_text -o locales/AISurveillant.pot --join-existing surveillance_ui/*.py
for lang in "en" "cs"; do
    if [ ! -d locales/$lang ]; then
        mkdir locales/$lang
    fi
    if [ ! -d locales/$lang/LC_MESSAGES ]; then
        mkdir locales/$lang/LC_MESSAGES
    fi      
    msginit --input=locales/AISurveillant.pot --locale=$lang --output=locales/$lang/LC_MESSAGES/AISurveillant.new.po --no-translator
    if [ -f locales/$lang/LC_MESSAGES/AISurveillant.po ]; then
        msgmerge -U locales/$lang/LC_MESSAGES/AISurveillant.po --backup=none locales/$lang/LC_MESSAGES/AISurveillant.new.po
        rm locales/$lang/LC_MESSAGES/AISurveillant.new.po
    else
        mv locales/$lang/LC_MESSAGES/AISurveillant.new.po locales/$lang/LC_MESSAGES/AISurveillant.po
    fi
    msgfmt -o locales/$lang/LC_MESSAGES/AISurveillant.mo locales/$lang/LC_MESSAGES/AISurveillant.po
done
rm locales/AISurveillant.pot
