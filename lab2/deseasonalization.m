jj_max=length(el_lo)/24;
%reuild as a matrix with day and time
for ii=1:24
    for jj=1:jj_max
        el_lo_mat(ii,jj)=el_lo((jj-1)*24+ii);
    end
end
% compute the mean load for each time of the day
el_lo_mean=mean(el_lo_mat');

% compute the desesonalized load 
for ii=1:24
    for jj=1:jj_max
        el_lo_des((jj-1)*24+ii)=el_lo_mat(ii,jj)-el_lo_mean(ii);
    end
end
