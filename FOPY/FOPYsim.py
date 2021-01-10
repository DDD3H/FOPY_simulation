# -*- coding: utf-8 -*-
###########################################################
### Choose the type of simulation(弾道or減速)
### 1=弾道落下 2=減速落下 3=Delay time
###########################################################

from FOPYparameters import *        # parameterの読み込み
from FOPYthrust import *            # thrustデータの読み込み
from FOPYprset import *
import pandas as pd
import numpy.linalg as LA

# IV  = [Ve1,Ve2,Ve3,Xe1,Xe2,Xe3,omg2,omg3,q1,q2,q3,q4]
Ve  = np.array([[Ve1],[Ve2],[Ve3]])            #地球座標系における機体速度ベクトル
# Xe  = [IV(4), IV(5), IV(6)]            #地球座標系における機体位置ベクトル
# omg = [0, IV(7), IV(8)]                #角速度
# q   = [IV(9), IV(10), IV(11), IV(12)]  #クォータニオン
#Winddata =
Winddata = np.loadtxt('winddata_1123.csv', delimiter=',', encoding='utf-8')
print(Winddata)
i = 1                                  #ステップ数
t = 0                                  #時間
###
for i in range(0, n):                 #1~nの計算
    t = i * dt                          # time [s]
    Aeb = np.array([[1-2*((q2**2)+(q2**2)),2*(q1*q2+q2*q2),2*(q2*q1-q2*q2)],
                    [2*(q1*q2-q2*q2),1-2*((q1**2)+(q2**2)),2*(q2*q2+q1*q2)],
                    [2*(q2*q1+q2*q2),2*(q2*q2-q1*q2),1-2*((q1**2)+(q2**2))]])



    # transformation matrix(earth frame --> body frame)(座標変換行列)(地上座標系を機体座標系へ)


    if t < tThrust:                                # 燃焼中の飛翔
        T       = thrust[i]                    # thrust [N]
        mdot    = (m0 - mf) / tThrust              # weight loss [kg/sec]
        m       = m0 - mdot * t                    # weight [kg]
        I       = I0 - (I0 - If) / tThrust * t         # moment of inertia [kgm^2]
        lcg     = lcg0 - (lcg0 - lcgf) / tThrust*t   # center of gravity [m]
    else:                                        # 燃焼終了後
        T       = 0                            # thrust [N]
        mdot    = 0                            # weight loss [kg/sec]
        m       = mf                           # weight [kg]
        I       = If                           # moment of inertia [kgm^2]
        lcg     = lcgf                         # center of gravity [m]


    #rho = 1.23-(0.11e-3)*Xe(3);                 # atmospheric density
                                                # Don't use this equation over 2km
    r0 = 6356766
    H = (r0*Xe3*0.001)/(r0+Xe3*0.001)       # ジオポテンシャル高度
    Temp = 15 - 6.5*H
    P = 101325 * (288.15/(Temp +273.15))**(-5.256)
    rho = (0.0034837*P)/(Temp+273.15)       # Don't use this equation over 11km
    # wind
    if WindModel == 1:
        if i == 0:           #ステップが1の時は高度0mなので仕方なく計測した風速そのまま利用
            Vw = np.array([[Vwaz*math.cos(Waz)],[Vwaz*math.sin(Waz)],[0]])
        else:
            Vw = np.array([[Vwaz*((Xe3/Zr)**(1/Cdv))*math.cos(Waz)],[Vwaz*((Xe3/Zr)**(1/Cdv))*math.sin(Waz)],[0]])                   #べき法則に基づいた計算
            print("hello")
    elif WindModel == 2:
        Vw = np.array([[Vwaz*math.cos(Waz)], [Vwaz*math.sin(Waz)], [0]])
    elif WindModel == 3:                                                # 統計風
        if i == 1:                                           # エラー回避用
            Vw = np.array([[0.1],[0.1],[0]])
        else:
            Xew = np.ceil(Xe3+0.5)
            VwazH = Winddata[int(Xew)-1,0]
            WazH = Winddata[int(Xew)-1,1]
            Vw = np.array([[VwazH*math.cos(WazH)],[VwazH*math.sin(WazH)],[0]])



    # airspeed [m/s](対気速度)
    Vab = Aeb @ (Ve - Vw)
    Va = LA.norm(Vab)
    # Ve:地球座標系(earth frame)における機体速度ベクトル
    # Vw:地球座標系(earth frame)における風速ベクトル

    # angle of attack & angle of sideslip(迎角&横滑り角)
    # alpha = asin(Vab(3)/Va);
    print(Vab)
    alpha = math.atan(Vab[2,0]/Vab[0,0])
    bet = math.asin(Vab[1,0]/Va)

    # drag & normal force & side force
    D = 0.5*rho*Va*Va*S*Cd
    Y = 0.5*rho*Va*Va*S*Cnalpha*math.sin(bet)
    N = 0.5*rho*Va*Va*S*Cnalpha*math.sin(alpha)
    # D:抗力(作用方向は軸力)
    # Y:機体軸(Y軸方向)に作用する力→side force
    # N:法線力

    # force @body frame
    Fe = np.array([[T-D],[-Y],[-N]])
    print(Xe)
    log_Xe[:,i] = Xe
    print(log_Xe)
    [xmax,tmax] = max(log_Xe[2,:])
    tmin = size(log_Xe[2,:])
    MP1 = log_Xe[2,tmax:tmin(0,1)
    MP2 = find(MP1>Hpara)
    tpara = size(MP2)*dt+tmax*dt;
    tdelay = tmax*dt+Dpara;

    # acceleration/velocity/position @earth-fixed frame
    if SIMULATION == 1: #弾道落下
        Ae = Aeb.T@ (Fe/m) - np.array([[0,0,g]])
        Ve = Ae * dt + Ve
        Xe = Ae*dt + Ve
    elif SIMULATION == 2: # 減速落下：下段の燃焼終了後かつ速度が負でパラ展開，終端速度に即達する
            if (Ve3 <= 0) and (t > tThrust) and (Xe3 > Hpara)
                Ae = np.array([][0],[0],[0]])
                Ve = np.array([[Vw[0,0]],
                               [Vw[0,1]],
                               []-Vpara1 * math.tanh(g*(t - tmax * dt)/Vpara1)]])
                Xe = Ve*dt+Xe
            elseif (Ve(3)<=0)&&(t>tThrust)&&(Xe(3)<Hpara)   # 2段パラ
                Ae = [0;0;0];
                Ve = [Vw[1]; Vw(2); -Vpara1+(Vpara1-Vpara2)*tanh(g*(t-tpara(1,2))/(Vpara1-Vpara2))];
                Xe = Ve*dt+Xe;
            else                                            # 上昇中
                Ae = Aeb'*(Fe./m)-[0;0;g];
                Ve = Ae*dt+Ve;
                Xe = Ve*dt+Xe;
            #end
        case 3                         # 減速落下：下段の燃焼終了後かつ速度が負でパラ展開，終端速度に即達する
            if (t>=tdelay)&&(t>tThrust)&&(Xe(3)>Hpara)
                Ae = [0;0;0];
                Ve = [Vw[1]; Vw(2); -Vpara1*tanh(g*(t-tmax*dt)/Vpara1)];
                Xe = Ve*dt+Xe;
            elseif (t>=tdelay)&&(t>tThrust)&&(Xe(3)<Hpara)   # 2段パラ
                Ae = [0;0;0];
                Ve = [Vw[1]; Vw(2); -Vpara1+(Vpara1-Vpara2)*tanh(g*(t-tpara(1,2))/(Vpara1-Vpara2))];
                Xe = Ve*dt+Xe;
            else                                            # 上昇中
                Ae = Aeb'*(Fe./m)-[0;0;g];
                Ve = Ae*dt+Ve;
                Xe = Ve*dt+Xe;
            #end
    #end

    Veab = norm(Ve);  #機速
    Aeab = norm(Ae);  #機速

    # coefficient
    Kj = -(Ip0/mp0+(lcg-lcgp)^2-(l-lcg)^2)*mdot;
    Ka = 0.5*rho*S*(Va^2)*(l^2)/2/Va*Cmq;
    # Kj:ジェットダンピング係数
    # Ka:空気力による減衰モーメント係数

    # angular velocity
    kt1 = -((lcp-lcg)*N+(Ka+Kj)*omg(2))/I;
    kt2 = -((lcp-lcg)*N+(Ka+Kj)*(omg(2)+kt1*dt/2))/I;
    kt3 = -((lcp-lcg)*N+(Ka+Kj)*(omg(2)+kt2*dt/2))/I;
    kt4 = -((lcp-lcg)*N+(Ka+Kj)*(omg(2)+kt3*dt))/I;

    kp1 = ((lcp-lcg)*Y+(Ka+Kj)*omg(3))/I;
    kp2 = ((lcp-lcg)*Y+(Ka+Kj)*(omg(3)+kp1*dt/2))/I;
    kp3 = ((lcp-lcg)*Y+(Ka+Kj)*(omg(3)+kp2*dt/2))/I;
    kp4 = ((lcp-lcg)*Y+(Ka+Kj)*(omg(3)+kp3*dt))/I;
    # kt:y軸周りの角速度
    # kp:z軸周りの角速度

    switch(SIMULATION)
        case 1
            if (Xe(3)<lLnchr)&&(t<tThrust)                    #ランチャに刺さってる時は回転しない
                omg = [0;0;0];
            else                                              #飛翔中ずっとルンゲクッタ
                omg = [0;
                       1/6*(kt1+2*kt2+2*kt3+kt4)*dt+omg(2);
                       1/6*(kp1+2*kp2+2*kp3+kp4)*dt+omg(3)];
            end

        case 2
            if (Xe(3)<lLnchr)&&(t<tThrust)                    #ランチャに刺さってる時は回転しない
                omg = [0;0;0];
            elseif (Ve(3)<=0)&&(t>tThrust)                    #減速落下中も回転しない
                omg = [0;0;0];
            else                                              #それ以外の時(上昇中)ルンゲクッタ
                omg = [0;
                       1/6*(kt1+2*kt2+2*kt3+kt4)*dt+omg(2);
                       1/6*(kp1+2*kp2+2*kp3+kp4)*dt+omg(3)];
            end
        case 3
            if (Xe(3)<lLnchr)&&(t<tThrust)                    #ランチャに刺さってる時は回転しない
                omg = [0;0;0];
            elseif (t>=tdelay)&&(t>tThrust)                    #減速落下中も回転しない
                omg = [0;0;0];
            else                                              #それ以外の時(上昇中)ルンゲクッタ
                omg = [0;
                       1/6*(kt1+2*kt2+2*kt3+kt4)*dt+omg(2);
                       1/6*(kp1+2*kp2+2*kp3+kp4)*dt+omg(3)];
            end
    end

# quaternion
qmat = [0 omg(3) -omg(2) 0;
        -omg(3) 0 0 omg(2);
        omg(2) 0 0 omg(3);
        0 -omg(2) -omg(3) 0];
q = (0.5*qmat*q*dt+q)/norm(q);

# attitude angle(姿勢角)
the = asin(Aeb(1,3));
psi = atan(Aeb(1,2)/Aeb(1,1));

# ランチクリア速度
if (norm(Xe)<lLnchr) && (t<tThrust)
   Vlc = Veab[1];
else
   Vlc = 0;
end

# log
log_t(1,i)     = t;
log_T(1,i)     = T;
log_m(1,i)     = m;
log_I(1,i)     = I;
log_lcg(1,i)   = lcg;
log_rho(1,i)   = rho;
log_Vw(:,i)    = Vw;
log_Vab(:,i)   = Vab;
log_Va(1,i)    = Va;
log_alpha(1,i) = alpha*180/pi;
log_bet(1,i)   = bet*180/pi;
log_D(1,i)     = D;
log_Y(1,i)     = Y;
log_N(1,i)     = N;
log_Fe(:,i)    = Fe;
log_Ae(:,i)    = Ae;
log_Ve(:,i)    = Ve;
log_Veab(1,i)  = Veab;
log_Aeab(1,i)  = Aeab;
log_Kj(1,i)    = Kj;
log_Ka(1,i)    = Ka;
log_omg(:,i)   = omg*180/pi;
log_q(:,i)     = q;
log_the(1,i)   = the*180/pi;
log_psi(1,i)   = psi*180/pi;
log_Vlc(1,i)   = Vlc;
#

if (real(log_Xe(3,i))<0)&&(log_t(1,i)>tThrust)
    break
end
end

# plot
figure
plot(log_t(1,:),real(log_Xe(1,:)),'r',log_t(1,:),real(log_Xe(2,:)),'b',...
    log_t(1,:),real(log_Xe(3,:)),'g')
xlabel('Time, t [sec]');
ylabel('Distance, Xe [m]');
legend('Xe[1]','Xe(2)','Xe(3)');

figure
plot(log_t(1,:),real(log_Ve(1,:)),'r',log_t(1,:),real(log_Ve(2,:)),'b',...
    log_t(1,:),real(log_Ve(3,:)),'g')
xlabel('Time, t [sec]');
ylabel('Velocity, Ve [m/s]');
legend('Ve[1]','Ve(2)','Ve(3)');

figure
plot(log_t(1,:),real(log_the(1,:)),'r',log_t(1,:),real(log_psi(1,:)),'b')
xlabel('Time, t [sec]');
ylabel('Attitude Angle [deg]');
legend('the','psi');

figure
plot(log_t(1,:),real(log_omg(2,:)),'r',log_t(1,:),real(log_omg(3,:)),'b')
xlabel('Time, t [sec]');
ylabel('Angular Velocity, [deg/s]');
legend('pitch (omg(2))','yaw (omg(3))');

figure
plot(log_t(1,:),real(log_alpha(1,:)),'r',log_t(1,:),real(log_bet(1,:)),'b')
xlabel('Time, t [sec]');
ylabel('Angle [deg]');
legend('Angle of Attack (alpha)','Angle of Sideslip (bet)');

figure
plot(log_t(1,:),real(log_D(1,:)),'r',log_t(1,:),real(log_N(1,:)),'b',...
    log_t(1,:),real(log_Y(1,:)),'g');
xlabel('Time, t [sec]');
ylabel('Force [N]');
legend('Drag (D)','Normal Force (N)','Side Force (Y)');

figure
plot(log_t(1,:),real(log_Vab(1,:)),'r',log_t(1,:),real(log_Vab(2,:)),'b',...
    log_t(1,:),real(log_Vab(3,:)),'g')
xlabel('Time, t [sec]');
ylabel('Air Speed, Va [m/s]');
legend('Vab[1]','Vab(2)','Vab(3)');

figure
plot3(real(log_Xe(1,:)),real(log_Xe(2,:)),real(log_Xe(3,:)));
xlabel('東西 [m]')
ylabel('北南 [m]')

figure
plot(log_t(1,:),real(log_Aeab(1,:)));
xlabel('Time, t [sec]');
ylabel('Acceleraition [m/s^2]');
grid on

# display
fprintf('ver1.6：MODE##d\n',SIMULATION)
fprintf('風向：#ddeg，風速#dm/s\n',WazDeg,Vwaz);
fprintf('下段燃焼時間：#.2fs，頂点到達時間#.2fs\n',tThrust,tmax*dt);
fprintf('最高到達高度：#fm\n',max(log_Xe(3,:)));
fprintf('最高対気速度：#fm/s\n',max(log_Va(1,:)));
fprintf('ロンチャ離脱速度：#fm/s\n',max(log_Vlc(1,:)));
fprintf('ダウンレンジ：#fm\n',norm(log_Xe(:,end-1)));
fprintf('ダウンレンジ角度：#fdeg\n',180/pi*atan(log_Xe(2,end-1)/log_Xe(1,end-1)));
