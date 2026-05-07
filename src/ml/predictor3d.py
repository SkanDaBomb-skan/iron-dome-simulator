import numpy as np

class Radar3D:
    def __init__(self, sigma=30, frequence=10):
        self.sigma=sigma;self.frequence=frequence
    def observer(self, t_v, x_v, y_v, z_v, duree=8.0):
        m=t_v<=duree;to=t_v[m];xo=x_v[m];yo=y_v[m];zo=z_v[m]
        ind=np.arange(0,len(to),max(1,int(1.0/(self.frequence*0.1))))
        t=to[ind];xr=xo[ind];yr=yo[ind];zr=zo[ind]
        return {'t':t,'x':xr+np.random.normal(0,self.sigma,len(t)),
                'y':np.maximum(yr+np.random.normal(0,self.sigma,len(t)),0),
                'z':zr+np.random.normal(0,self.sigma,len(t)),'n':len(t)}

class UKF3D:
    def __init__(self, sigma_mesure=30.0):
        self.g=9.81;self.sm=sigma_mesure
        self.H=np.zeros((3,7));self.H[0,0]=1;self.H[1,1]=1;self.H[2,2]=1
        self.R=np.eye(3)*sigma_mesure**2
    def _f(self,s,dt):
        x,y,z,vx,vy,vz,k=s;v=np.sqrt(vx**2+vy**2+vz**2)
        if v>0 and k>0:ax=-k*v*vx;ay=-self.g-k*v*vy;az=-k*v*vz
        else:ax=0;ay=-self.g;az=0
        return np.array([x+vx*dt+.5*ax*dt**2,y+vy*dt+.5*ay*dt**2,z+vz*dt+.5*az*dt**2,vx+ax*dt,vy+ay*dt,vz+az*dt,k])
    def _jac(self,s,dt):
        x,y,z,vx,vy,vz,k=s;v=np.sqrt(vx**2+vy**2+vz**2);F=np.eye(7);F[0,3]=dt;F[1,4]=dt;F[2,5]=dt
        if v>1e-6 and k>0:
            dvx=vx/v;dvy=vy/v;dvz=vz/v
            F[0,3]+=.5*(-k*(v+vx*dvx))*dt**2;F[0,4]=.5*(-k*vx*dvy)*dt**2;F[0,5]=.5*(-k*vx*dvz)*dt**2;F[0,6]=.5*(-v*vx)*dt**2
            F[1,3]=.5*(-k*vy*dvx)*dt**2;F[1,4]+=.5*(-k*(v+vy*dvy))*dt**2;F[1,5]=.5*(-k*vy*dvz)*dt**2;F[1,6]=.5*(-v*vy)*dt**2
            F[2,3]=.5*(-k*vz*dvx)*dt**2;F[2,4]=.5*(-k*vz*dvy)*dt**2;F[2,5]+=.5*(-k*(v+vz*dvz))*dt**2;F[2,6]=.5*(-v*vz)*dt**2
            F[3,3]+=(-k*(v+vx*dvx))*dt;F[3,4]=(-k*vx*dvy)*dt;F[3,5]=(-k*vx*dvz)*dt;F[3,6]=(-v*vx)*dt
            F[4,3]=(-k*vy*dvx)*dt;F[4,4]+=(-k*(v+vy*dvy))*dt;F[4,5]=(-k*vy*dvz)*dt;F[4,6]=(-v*vy)*dt
            F[5,3]=(-k*vz*dvx)*dt;F[5,4]=(-k*vz*dvy)*dt;F[5,5]+=(-k*(v+vz*dvz))*dt;F[5,6]=(-v*vz)*dt
        return F
    def filtrer(self,t_obs,x_obs,y_obs,z_obs):
        if len(t_obs)<2:return np.array([[x_obs[0],y_obs[0],z_obs[0],0,0,0,0.0001]])
        dt=t_obs[1]-t_obs[0]
        self.x=np.array([x_obs[0],y_obs[0],z_obs[0],(x_obs[1]-x_obs[0])/dt,(y_obs[1]-y_obs[0])/dt,(z_obs[1]-z_obs[0])/dt,0.00008])
        self.P=np.diag([self.sm**2]*3+[(self.sm/dt)**2]*3+[0.0003**2])
        Q=np.diag([25]*3+[625]*3+[1e-10]);etats=[]
        for j in range(len(t_obs)):
            F=self._jac(self.x,dt);self.x=self._f(self.x,dt);self.P=F@self.P@F.T+Q
            self.x[6]=np.clip(self.x[6],1e-5,1e-3)
            z=np.array([x_obs[j],y_obs[j],z_obs[j]]);S=self.H@self.P@self.H.T+self.R
            K=self.P@self.H.T@np.linalg.inv(S);self.x=self.x+K@(z-self.H@self.x)
            self.x[6]=np.clip(self.x[6],1e-5,1e-3);self.P=(np.eye(7)-K@self.H)@self.P;etats.append(self.x.copy())
        return np.array(etats)
