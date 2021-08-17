[toc]



# 配置文件

## .ssh/config

```
Host *
    ServerAliveInterval 30
    ServerAliveCountMax 200 
StrictHostKeyChecking no
Host zhuk
    HostName 222.30.62.19
    Port 22
    User zhuk
Host www 
    HostName 222.30.62.19
    Port 22
    User yangserver
Host ibm 
    HostName 222.30.48.146
    Port 22
    User sry 
Host gpu 
    HostName 222.30.62.7
    Port 22
    User sry 
Host dell
    HostName 10.2.0.4
    Port 9920
    User zzp 
Host meng
    HostName 139.198.16.150
    Port 9002
    User rsong
Host long
    HostName 139.198.16.150
    Port 9001
    User rsong
Host hps
    HostName 222.30.62.19
    Port 22
    User sry
    ProxyCommand ssh -p 22 sry@222.30.48.146 -W %h:%p
```



## .bash_profile

```bash
# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
        . ~/.bashrc
fi

# User specific environment and startup programs

#PATH=$PATH:$HOME/.local/bin:$HOME/bin:$HOME/opt/local/bin
PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH
```

## .bashrc

```shell
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

#
# specific aliases by ruiyang
#
# safe and convenient
alias cp="cp -i"
alias mv="mv -i"
alias rm="rm -i"
alias ll="ls -lshtAF"
alias la="ls -AF"
alias l="ls -CF"
# count file, link, folder and n. of columns
alias n-='ls -l | grep '^-' | wc -l'
alias nl='ls -l | grep '^l' | wc -l'
alias nd='ls -l | grep '^d' | wc -l'
alias nf='count() { cat $1 | awk -F $2 "{print NF}"; }; count'
#alias qn='kill_by_name() { ps -aux |grep sry |grep $1 | awk -F " " "{print(\$2)}"; }; kill_by_name'
# some more
alias q='exit'
alias qs='qzy | grep sry'
alias qsa='qzy | grep sry_'
alias last='last | tac'
alias bio='conda activate bio'
# server
alias www='ssh www'
alias zhuk='ssh zhuk'
alias gpu='ssh gpu'
alias ibm='ssh ibm'
alias dell='ssh dell'

#alias hpj="ssh hps -N -f -L localhost:8889:222.30.62.19:8888"
#alias hpzj="ssh hps -N -f -L localhost:9001:222.30.62.19:9001"

# tmux
export LD_LIBRARY_PATH=/public/home/sry/opt/libevent/lib:$LD_LIBRARY_PATH
alias tat='tmux attach -t'
alias tns='tmux new -s'
# PS1 color
PS1='\[\e[36;1m\][\u@\h:\w]\$ \[\e[0m\]'


# Amber
export AMBERHOME=/public/home/sry/opt/STRUM/amber14
#source /public/home/sry/opt/STRUM/amber20/amber.sh

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
# conda config --set auto_activate_base false
__conda_setup="$('/public/home/sry/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/public/home/sry/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/public/home/sry/opt/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/public/home/sry/opt/miniconda3/bin:$PATH"
    fi
fi
conda activate bio
unset __conda_setup
# <<< conda initialize <<<
```

## .vimrc

```
hi comment ctermbg=4 ctermfg=6
set nu
set nowrap
```



## .condarc

```
auto_activate_base: false
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

## .tmux.conf

```
# ~/.tmux.conf

# unbind default prefix and set it to ctrl-a
unbind C-b 
set -g prefix C-a 
bind C-a send-prefix

# make delay shorter
set -sg escape-time 0

# buffer size
set -g history-limit 10000

# clock
#setw -g clock-mode-colour colour64 #green

#### key bindings ####

# reload config file
bind r source-file ~/.tmux.conf \; display ".tmux.conf reloaded!"

# quickly open a new window
bind N new-window

# synchronize all panes in a window
#bind y setw synchronize-panes

# pane movement shortcuts (same as vim)
bind -n M-Left select-pane -L
bind -n M-Down select-pane -D
bind -n M-Up select-pane -U
bind -n M-Right select-pane -R

# window movement shortcuts
bind -n C-Left previous-window
bind -n C-Right next-window

# Easier window split keys
bind-key v split-window -h
bind-key h split-window -v
# enable mouse support for switching panes/windows
#set -g mouse-utf8 on
set -g mouse on

########## Config mouse mode on different version of tmux
##lt_2.1.conf contains
#set -g mode-mouse on
#set -g mouse-resize-pane on
#set -g mouse-select-pane on
#set -g mouse-select-window on

##gt_2.1.conf contains
#set -g mouse-utf8 on
#set -g mouse on


#### copy mode : vim ####

# set vi mode for copy mode
setw -g mode-keys vi

# copy mode using 'Esc'
unbind [
bind Escape copy-mode

# start selection with 'space' and copy using 'y'
#bind -t vi-copy 'y' copy-selection

# paste using 'p'
unbind p
bind p paste-buffer

# status bar position
set-option -g status-position top
```

# 自定义脚本

fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh

## qzy

```perl
#!/usr/bin/perl

$o=$ARGV[0];

%S=(
    'R'=>'R',
    'Q'=>'=',
    'S'=>'%',
    'E'=>'-',
    );

my %all=();
my @rst=`/usr/local/bin/qstat`;
foreach my $r(@rst)
{
    next if($r !~ /^\d+/);
    if($r =~/\S+\s+\S+\s+(\S+)/)
    {
	$all{$1}++;
	#print "$r";
	#print "$1\n";
    }
}

my @keys= sort{$all{$b} <=> $all{$a}} keys %all;

#print "\n@keys\n";

@gg=(zhanglabs);

$member{zhanglab} .= "zhanglabs ";
foreach my $k(@keys)
{
    if($k ne "zhanglabs" && $k ne "yangji" && $k ne "zhng")
    {
	#print "$k\n";
	push(@gg, $k);
	$member{zhanglab} .= "$k ";
    }    
}
push(@gg, "yangji");
$member{zhanglab} .= "yangji";

#print "\n@gg\n";
#exit;
@labs=qw(
	 zhanglab
	 other
	 );



$member{other}="
lyzhang
jdpoisso
";

pos42:;
#@lines=`/usr/pbs/bin/qstat -f`;
@lines=`/usr/local/bin/qstat -f`;
$nl=@lines;
if($nl <10){
    printf "no job\n";
	exit;
    sleep(10);
    goto pos42;
}

$I=0;
$nn=0;
$nc=0;
$nn_sub=0;
$nr_all=0;
$n_yzhang=0;
$nr_group=0;
$nsub_zhang=0;
undef %nj;
foreach $line(@lines){
    $I++;
    $id=$1 if($line=~/Job Id\:\s*(\d+)/);
    $jobname=$1 if($line=~/Job_Name =\s*(\S+)/);
    $status=$1 if($line=~/job_state =\s*(\S+)/);
    $runtime=$1 if($line=~/resources_used.walltime =\s*(\S+)/);
    $qtime=$1 if($line=~/qtime =\s*(.+)/);
    $user=$1 if($line=~/Job_Owner =\s*(\S+)\@/);
    $walltime=$1 if($line=~/Resource_List.walltime =\s*(\S+)/);
    if($line=~/queue =\s*(\S+)/){
	$q=$1;
	if($q eq "urgent"){
	    $q="UG";
	}
    }
    $mk=1 if($line=~/etime/); #end of the job statement
    #######
    #if($line=~/exec_vnode =/){
    if($line=~/exec_host =/){
	$tmp1="";
	for($j=$I;$j<=$I+100;$j++){
	    $tmp=$lines[$j-1];
	    $tmp=~s/^\s+//mg;
	    $tmp=~s/\n//mg;
	    goto pos1 if($tmp=~/Hold_Type/);
	    goto pos1 if($tmp=~/Join_Path/);
	    $tmp1 .= $tmp;
	}
      pos1:
	@ll=split('\)',$tmp1);
	foreach $l(@ll){
	    if($l=~/compute-(\d+-\d+)\// || $l=~/node(\d+)/){		
		$node="node$1";
		$nn++;
		### for print
		$nc++;
		$comp{$nc}=$node;
		$n_comp{$nc}=1;
		###
	    }
	}
	#printf "$tmp1\n";
	#printf "id=$id, nn=$nn\n";
    }
    $nn=0 if($status ne "R");
    if($nn>1){
	$nn_sub=$nn;
    }else{
	$nn_sub=1;
    }
    if($line!~/\S+/ && $mk==1){
	#printf "-------out-------\n";
	#goto pos6 if($status eq "E"); #ngelect jobs with "-"
	$nr_all+=$nn;
	foreach $g(@gg){
	    if($user eq $g){ # Zhang-lab
		$nr_group+=$nn;
		$ng_r{$g}+=$nn;
		$ng_t{$g}+=$nn_sub;
		if($q=~/UG/){
		    $nr_high{$g}+=$nn;
		    $nt_high{$g}+=$nn_sub;
		}elsif($q=~/cas/){
		    $nr_casp{$g}+=$nn;
		    $nt_casp{$g}+=$nn_sub;
		}else{
		    $nr_low{$g}+=$nn;
		    $nt_low{$g}+=$nn_sub;
		}
		if($jobname=~/S2/){
		    $n_server+=$nn;
		}
	    }
	}
	foreach $lab(@labs){
	    if($member{$lab}=~/$user/){
		$n_lab{$lab}+=$nn;
		$n_lab_sub{$lab}+=$nn_sub;
		goto pos3;
	    }
	}
	$n_missuser++;
	$missuser{$n_missuser}=$user;
      pos3:;
	$node="" if($status ne "R" && $status ne "S" && $status ne "E");
	if(length $o >0 && $o ne "node"){
	    if($o eq "zhanglab"){
		if($q ne "urgent_zhanglab" && $q ne "zhanglab"){
		    goto pos6;
		}
	    }elsif($o eq "casp"){
		if($q ne "casp"){
		    goto pos6;
		}
	    }elsif($o eq "def"){
		if($q ne "default"){
		    goto pos6;
		}
	    }elsif($o eq "UG"){
		if($q ne "UG"){
		    goto pos6;
		}
	    }elsif($o eq "R"){
		if($status ne "R"){
		    goto pos6;
		}
	    }else{
		if($user ne $o){
		    goto pos6;
		}
	    }
	}
	#printf "nc=$nc\n";
	if($nc>0){
	    for($i1=1;$i1<=$nc;$i1++){
		printf "%7s %3s %7s %7s(%2d/%4d) %1s %25s %9s %9s %15s\n",
		$id,substr($q,0,3),substr($user,0,7),$comp{$i1},
		$n_comp{$i1},$nr_group,
		$S{$status},substr($jobname,0,25),
		$runtime,$walltime,
		substr($qtime,4,15);

		####### for node statistics ###############
		if($node=~/\S/){
		    $nj{$node}++;
		    $job{$node,$nj{$node}}=$jobname;
		    $who{$node,$nj{$node}}=$user;
		}
	    }
	}else{
	    printf "%7s %3s %7s %7s(%2d/%4d) %1s %20s %9s %9s %15s\n",
	    $id,substr($q,0,3),substr($user,0,7),$node,
	    $nn,$nr_group,
	    $S{$status},substr($jobname,0,20),
	    $runtime,$walltime,
	    substr($qtime,4,15);
	    
	    ####### for node statistics ###############
	    if($node=~/\S/){
		$nj{$node}++;
		$job{$node,$nj{$node}}=$jobname;
		$who{$node,$nj{$node}}=$user;
	    }
	}
	
	pos6:;
	
	$id="";;
	$user="";
	$node="";
	$nn=0;
	$nc=0;
	$status="";
	$jobname="";
	$runtime="";
	$walltime="";
	$qtime="";
	$mk="";
    }
}
#$N_high=160;
#$N_low=116; #58*2=116, 58*3=174 #////
#printf "\n my_high_max= 0 my_low_max= 500 my_tot_max= 0 n_tot_max= 1000 n_tot_casp= 0\n";
printf "\n--------      n_urgent  n_default    n_casp      n_all\n";
$ng_rt=0; 
$ng_tt=0;
foreach $g(@gg){
    $ng_rt+=$ng_r{$g};
    $ng_tt+=$ng_t{$g};
    $nr_high_all+=$nr_high{$g};
    $nr_low_all+=$nr_low{$g};
    $nr_casp_all+=$nr_casp{$g};
    $nt_high_all+=$nt_high{$g};
    $nt_low_all+=$nt_low{$g};
    $nt_casp_all+=$nt_casp{$g};
	next if($ng_t{$g}==0);
    printf "%10s\_ %4d(%4d) %4d(%4d) %4d(%4d) %4d(%4d)\n",
    $g,
    $nt_high{$g},$nr_high{$g},
    $nt_low{$g},$nr_low{$g},    
    $nt_casp{$g},$nr_casp{$g},
    $ng_t{$g},$ng_r{$g};
}
printf "%10s\_ %4d(%4d) %4d(%4d) %4d(%4d) %4d(%4d)\n",
    '_all',
    $nt_high_all,$nr_high_all,
    $nt_low_all,$nr_low_all,    
    $nt_casp_all,$nr_casp_all,
    $ng_tt,$ng_rt;

printf "------\n";
$n_sub=0;
foreach $lab(@labs){
    $n_sub+=$n_lab_sub{$lab};
    printf "%8s %5d %5d\n",$lab,$n_lab{$lab},$n_lab_sub{$lab};
}
printf "%8s %5d %5d\n","ALL",$nr_all,$n_sub;

if($n_missuser>0){
    for($i=1;$i<=$n_missuser;$i++){
	printf "$i $missuser{$i} is not in a lab\n";
    }
}
############ count nodes ===================>
$n_free=0;
for($i=1;$i<=17;$i++){
	next if($i==13);
    $node="node$i";
    if($o eq "node"){
        $tmp=24-$nj{$node};
        if($tmp==24){
            printf "%5s: down",$node;
        }elsif($tmp==0){
            printf "%5s: -- ",$node;
        }else{
            printf "%5s: %2d ",$node,$tmp;
        }
    }
    if($o eq "node"){
        $n1=0;
        for($k=1;$k<=$nj{$node};$k++){
            printf " %5s(%4s)",
            substr($job{$node,$k},0,5),
            substr($who{$node,$k},0,4);
            $n1++;
            if($n1>=5){
                printf "\n";
                printf "%10s";
                $n1=0;
            }
        }
        printf "\n";
    }
    $n_node_tot++;
    #printf "$node - $nj{$node}\n";
    if($nj{$node}<1){ #node down
        $n_down++;
    }else{     #node up
        $n_up++;
        $n_free+=24-$nj{$node};
    }
}

printf "\n#node_tot= %3d, #node_on= %3d, #node_down= %3d\n",
    $n_node_tot,$n_up,$n_down;
printf "#job_cap= %4d, #job_max= %4d,  #job_run= %4d, #job_idle= %4d\n",
    $n_node_tot*24,$n_up*24,$n_up*24-$n_free,$n_free;

$date=`/bin/date`;
print "$date\n";

exit();
```



## delQ

```perl
#!/usr/bin/perl -w
my $user = `whoami`;
$user =~ s/\n//;
my $len_argv = @ARGV;
if ($len_argv == 0){ 
    `qselect -u $user | xargs qdel`;
    exit(0);
}
else{
    my $tag_pre = $ARGV[0];
    my @jobs = `qstat -u $user | grep $tag_pre`;
    foreach my $job(@jobs){
        $job =~ s/\n//;
        if ($job =~ /^(\d+)/){
            print "delete job [$job]\n";
            `qdel $1`;
        }   
    }   
}
```

## bak

```python
#!/usr/bin/env python
'-sry'
import os, time, argparse
def shell(cmd):
    res=os.popen(cmd).readlines()[0].strip()
    return res 
homedir = shell('echo $HOME')
parser = argparse.ArgumentParser()
parser.description='****** Backup file as a conpressed tar ball to BAKDIR with TIME stamp. ******'
parser.add_argument('FILEDIR',help='file for backup, default is "."',type=str,default='.')
parser.add_argument('-b','--bakdir',help='backupdir for storage.',type=str,default='%s/bak_share'%homedir)
parser.add_argument('-s','--scpserver',help='bakdir of scp server for storage.',type=str,default='gpu:~/bak_share/hp/')
args = parser.parse_args()
filedir = args.FILEDIR
if args.bakdir:
    bakdir = args.bakdir
    if bakdir[-1] == '/':
        bakdir = bakdir[:-1]
if not os.path.exists(bakdir):
    os.makedirs(bakdir)
if args.scpserver:
    scpserver = args.scpserver
filename = filedir.split('/')[-1]
if filename == '': 
    filename = filedir.split('/')[-2]
os.system("echo '---------- BEGIN ----------' | tee -a %s/bak.log"%bakdir)
os.system("echo 'pwd is:' `pwd` >> %s/bak.log"%bakdir)
os.system("echo 'Backup begin at: ' `date +%Y-%m-%d,%H:%M:S` >> "+"%s/bak.log"%bakdir)
time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
cmd="tar -czvf %s/"%bakdir + filename + "_" + time_stamp + ".tar.gz " + filedir +" >> %s/bak.log"%bakdir
os.system("echo 'tar -czvf %s/'"%bakdir + filename + "_" + time_stamp + '.tar.gz ' + filedir)
os.system("echo %s >> %s/bak.log"%(cmd,bakdir))
os.system(cmd)
try:
    bakfiledir = bakdir + '/' + filename + "_" + time_stamp + ".tar.gz"
    os.system("echo '---------- scp ----------' | tee -a %s/bak.log"%bakdir)
    scpcmd = "scp %s %s"%(bakfiledir,scpserver)
    os.system("echo %s >> %s/bak.log"%(scpcmd,bakdir))
    os.system(scpcmd)
    os.system("echo 'Backup end at: ' `date +%Y-%m-%d,%H:%M:S` >> "+"%s/bak.log"%bakdir)
    os.system("echo '---------- END ----------'")
except:
    print('[ERROR] scp error!')
```



## du_node

```perl
#! /usr/bin/perl -w
use strict;

my $user=`whoami`;
chomp($user);

for(my $i=1; $i<=17; $i++){
    my $j="node$i";
    print "Cleaning for node $j\n";    
    print `ssh $j du -h /tmp/$user`;
    #`ssh $j find /tmp/$user/ -delete`;
    #print "done!\n";
}
```

# git

## 恢复

git log --pretty=oneline   提交历史
git reflog                 命令历史
git reset --hard HEAD^
git reset --hard 1094a
git reset HEAD <file>      以版本库最新版本恢复file，此时的缓存区也会清空
git checkout -- <file>     撤销工作区修改的file

## 分支

git branch                                      查看分支
git branch -vv                                  查看分支和远程对应关系
git branch <name>                               创建分支
git checkout <name> or git switch <name>        切换分支
git checkout -b <name> or git switch -c <name>  创建并切换分支
git branch -d <branch>                          删除一个分支

## merge

git merge dev     把dev分支的工作成果合并到当前分支上

## 远程库

ssh-keygen -t rsa -C "youremail@example.com"

git remote add origin git@github.com:michaelliao/learngit.git
git remote -v
git remote rm origin 删除关联的远程库

# jupyter

## kernel

```shell
## 安装kernel
(your-venv)$ conda install ipykernel

## list kernel
jupyter kernelspec list

## 安装kernel
(your-venv)$ ipython kernel install --name "local-venv" --user

## 更改kernel名字
# 1) Use $ jupyter kernelspec list to see the folder the kernel is located in
# 2) In that folder, open up file kernel.json and edit option "display_name"
```

## 插件



## jupyter_config.py 

```
c.NotebookApp.password = u'sha1:b5b88265812e:b1b828628ccea3d959bce1a8047a7f467e24cbb4'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8001
c.NotebookApp.open_browser = False
c.NotebookApp.allow_origin = '*'
c.NotebookManager.notebook_dir = '/nfs/project/songruiyang'
```

## nb

```shell
#!/bin/bash
nohup sh -c "CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) /home/luban/miniconda3/bin/jupyter notebook --config /home/luban/jupyter_config.py" > jupyter.log 2>&1 &
```

