# >>> conda initialize >>>
__conda_setup="$('$HOME/.pyenv/versions/miniforge3-4.10/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/.pyenv/versions/miniforge3-4.10/etc/profile.d/conda.sh" ]; then
        . "$HOME/.pyenv/versions/miniforge3-4.10/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/.pyenv/versions/miniforge3-4.10/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
