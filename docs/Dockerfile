FROM ubuntu:focal

WORKDIR /var/local

RUN apt-get update && apt-get install -y perl wget libfontconfig1 && \
    wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh && \
    apt-get clean

ENV PATH="${PATH}:/root/bin"

RUN fmtutil-sys --all

# Update TeX Live before installing packages
RUN tlmgr update --self && tlmgr update --all

RUN tlmgr install fancyhdr \
    algorithmicx \
    algorithms \
    appendix \
    biblatex \
    caption \
    chktex \
    chngcntr \
    csquotes \
    diagbox \
    enumitem \
    fontawesome \
    fontaxes \
    jknapltx \
    lastpage \
    latexindent \
    lipsum \
    listings \
    marvosym \
    opensans \
    orcidlink \
    pgf \
    preprint \
    synctex \
    tabu \
    texcount \
    threeparttable \
    titlesec \
    todonotes \
    pict2e \
    lineno \
    setspace