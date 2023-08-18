\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ibireprt}
\RequirePackage{kvoptions}
\DeclareBoolOption[false]{nobcorignoretitle}
\DeclareBoolOption[false]{grayscalebody}
\DeclareBoolOption[false]{grayscaletitle}
%\DeclareStringOption[scrbook]{baseclass}
\ProcessKeyvalOptions*
%\LoadClassWithOptions{\stsreprt@baseclass}
\LoadClassWithOptions{scrreprt}
%\RequirePackage[automark,headsepline]{scrpage2}
\RequirePackage{graphicx}
\RequirePackageWithOptions{xcolor}
\RequirePackage{titlepage}
\RequirePackage{setspace}
\TitlePageStyle{TUHH-IBI}
\RequirePackage{fancyhdr}

%---------------------------------------------------------------------------------------------------
% Kopf- und Fußzeilen
%---------------------------------------------------------------------------------------------------
%\pagestyle{scrheadings}
%\clearscrheadings
%\clearscrplain
%\clearscrheadfoot
%\ohead{\pagemark}
%\ihead{\headmark}

%\pagestyle{fancy}
%\lhead{\leftmark}
\fancyhead[OR]{\thepage}% ungerade Seiten, rechts \thepage
\fancyhead[OL]{\leftmark}% ungerade Seiten, links
\fancyhead[ER]{\leftmark}% gerade Seiten, rechts
\fancyhead[EL]{\thepage}% gerade Seiten, links \thepage
\cfoot{}
\pagestyle{fancy}

% 1.5 Zeilenabstand
%\onehalfspacing

%\newcommand{\twosideheadings}{\rehead{\leftmark}\lohead{\rightmark}}
%\newcommand{\onesideheadings}{\ihead{\leftmark}\chead{}}
%\ifstsreprt@grayscalebody
%\ohead{\setlength{\unitlength}{0.5cm}\begin{picture}(1,0)\put(0,0){\includegraphics[width=0.5cm]{logo/logo_sts_letters_gray}}\end{picture}}
%\else
%\ohead{\setlength{\unitlength}{0.5cm}\begin{picture}(1,0)\put(0,0){\includegraphics[width=1cm]%{logo/logo_tuhh_uke}}\end{picture}}
%\fi

%\makeatletter
%\if@twoside\twosideheadings \else \onesideheadings \fi
%\makeatother

%\ifx\chapterpagestyle\undefined\else\renewcommand*{\chapterpagestyle}{scrheadings}\fi
%\ifx\partpagestyle\undefined\else\renewcommand*{\partpagestyle}{scrheadings}\fi
%\ifx\indexpagestyle\undefined\else\renewcommand*{\indexpagestyle}{scrheadings}\fi
%\ifx\chapterheadstartvskip\undefined\else\renewcommand{\chapterheadstartvskip}{\vspace *{-\baselineskip }}\fi

\endinput



 


