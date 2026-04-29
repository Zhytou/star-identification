# SAO Star Catalogue

SAO星表，是史密松天体物理台星表（Smithsonian Astrophysical Observatory Star Catalog）的简称。它在1966年由美国史密森天文台为根据照相确定人造卫星位置的要求编制而成。

目前，网上能找到的SAO数据主要来自两个网站，分别是[哈佛大学天文望远镜中心 Harvard University Telescope Data Center](http://tdc-www.harvard.edu/catalogs/sao.html)和[美国国家航空航天局-高能天体物理科学档案研究中心 NASA's High Energy Astrophysics Science Archive Research Center(HEASARC)](https://heasarc.gsfc.nasa.gov/W3Browse/star-catalog/sao.html)下面对从这两处下载得到的原始SAO星表数据格式简单说明，方便后续视星等、双星和匀化处理。

## SAO From TDC

> raw_cata/SAO_b1950.binary

**Header Format**:

The first 28 bytes of both SAO and SAOra contain the following information:
Integer*4 STAR0=0   Subtract from star number to get sequence number
Integer*4 STAR1=1   First star number in file
Integer*4 STARN=258996  Number of stars in file
Integer*4 STNUM=1   0 if no star i.d. numbers are present
                    1 if star i.d. numbers are in catalog file
                    2 if star i.d. numbers are  in file
Logical*4 MPROP=t   True if proper motion is included
                    False if no proper motion is included
Integer*4 NMAG=1    Number of magnitudes present
Integer*4 NBENT=32  Number of bytes per star entry

**Entry Format**:

Each catalog entry in SAO and SAOra contains 32 bytes with the following information:

Real*4 XNO      Catalog number of star
Real*8 SRA0     B1950 Right Ascension (radians)
Real*8 SDEC0    B1950 Declination (radians)
Character*2 IS  Spectral type (2 characters)
Integer*2 MAG   V Magnitude * 100
Real*4 XRPM     R.A. proper motion (radians per year)
Real*4 XDPM     Dec. proper motion (radians per year)

**Sample Entries**:

The first and last entries in the SAO catalog file are:

SAO B1950 Equatorial Coordinates  Magnitude  run  8/ 2/1993
     1   0  00  05.097    82  41  41.82     7.20  A0
258996  23  54  51.661   -82  26  52.62     5.70  K0
The first and last entries in the RA-sorted SAOra catalog file are:

SAOra B1950 Equatorial Coordinates  Magnitude  run  8/ 2/1993
147051   0  00  00.156   -10  59  43.79     8.90  K2
255628  23  59  59.310   -61  40  13.53     9.80  G0

## SAO From HEASARC

> raw_cata/SAO_j2000.dat

**Line Format**:

Each line of the file is one entry of the star catalogue, and the position of each variable in the line is as follows.

- star id: 0-6
- visual magnitude: 80-84
- right ascension(j2000 in radians): 183-193
- declination(j2000 in radians): 193-204
