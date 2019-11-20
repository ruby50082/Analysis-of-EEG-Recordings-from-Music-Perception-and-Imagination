for p=1:9 
    if p==1
        load openMIIR-256hz/P01-24-1-epo.mat
    elseif p==2
        load openMIIR-256hz/P04-24-1-epo.mat
    elseif p==3
        load openMIIR-256hz/P05-24-1-epo.mat
    elseif p==4
        load openMIIR-256hz/P06-24-1-epo.mat
    elseif p==5
        load openMIIR-256hz/P07-24-1-epo.mat
    elseif p==6
        load openMIIR-256hz/P11-24-1-epo.mat
    elseif p==7
        load openMIIR-256hz/P12-24-1-epo.mat
    elseif p==8
        load openMIIR-256hz/P13-24-1-epo.mat
    elseif p==9
        load openMIIR-256hz/P14-24-1-epo.mat
    end
    
    dir='/Users/ruby/Documents/大三上/專題/data/5s/24/'; %第幾首歌
    n=numel(data(1,1,:));   %有幾個元素
    index=1;
    flag=0;
    data_num=256*5;
    for j=1:5
        for i=1:4
            n1=(i-1)*128+1;
            if (n1+(data_num-1))>n
                n2=n;
                n1=n2-(data_num-1);
                flag=1;
                break;
            else
                n2=n1+(data_num-1);
            end
            x=data(j,:,n1:n2);
            if p==1
                filename = strcat(dir,'P01-trail24-',num2str(index),'.mat'); %第幾個人 第幾首歌 切第幾段
            elseif p==2
                filename = strcat(dir,'P04-trail24-',num2str(index),'.mat');
            elseif p==3
                filename = strcat(dir,'P05-trail24-',num2str(index),'.mat');
            elseif p==4
                filename = strcat(dir,'P06-trail24-',num2str(index),'.mat');
            elseif p==5
                filename = strcat(dir,'P07-trail24-',num2str(index),'.mat');
            elseif p==6
                filename = strcat(dir,'P11-trail24-',num2str(index),'.mat');
            elseif p==7
                filename = strcat(dir,'P12-trail24-',num2str(index),'.mat');
            elseif p==8
                filename = strcat(dir,'P13-trail24-',num2str(index),'.mat');
            elseif p==9
                filename = strcat(dir,'P14-trail24-',num2str(index),'.mat');
            end
            save(filename,'x');
            index=index+1;
        end
        if flag==1
            break
        end
    end
end