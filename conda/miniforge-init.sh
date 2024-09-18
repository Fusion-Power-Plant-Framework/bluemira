# >>> conda initialize >>>
__conda_setup="$('$HOME/miniforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniforge/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniforge/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
