set autoindent
set background=dark
set colorcolumn=80,100,120,200
set cursorline
set hidden
set hlsearch
set ignorecase
set incsearch
set laststatus=2
set nocompatible
set number
set showmode
set statusline=%f\ %m%=%{&fileencoding?&fileencoding:&encoding}\ \|\ %{&fileformat}\ \|\ %p%%\ \|\ %l:%c
set viminfo=

set expandtab
set softtabstop=4
set shiftwidth=4
" set cursorcolumn

filetype plugin indent on
syntax on

if v:version >= 901
  " colorscheme habamax
  " colorscheme quiet
  " colorscheme retrobox
  colorscheme lunaperche
endif

command! Q q
command! W w
command! WQ wq
command! Wq wq
