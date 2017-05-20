#...................................................................
import ctypes
import numpy as np
import scipy.fftpack as fftpack
import h5py
import time
import sys
from scipy.special import laguerre
from scipy.special import genlaguerre
from scipy.special import legendre

#from pyfft.cuda import Plan
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.driver as cuda
import pycuda.autoinit

import pycuda.reduction as reduction
from pycuda.elementwise import ElementwiseKernel


from pycuda.compiler import SourceModule


#.................................................................
#            FAFT library
#.................................................................

DIR_BASE_FAFT = "/home/rcabrera/Documents/source/Python/PyWignerCUDA2/FAFT/"

# FAFT 64-points
_faft128_4D = ctypes.cdll.LoadLibrary( DIR_BASE_FAFT+'FAFT128_4D_Z2Z.so' )
_faft128_4D.FAFT128_4D_Z2Z.restype = int
_faft128_4D.FAFT128_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double ]

cuda_faft64 = _faft128_4D.FAFT128_4D_Z2Z


# FAFT 128-points
_faft256_4D = ctypes.cdll.LoadLibrary( DIR_BASE_FAFT+'FAFT256_4D_Z2Z.so' )
_faft256_4D.FAFT256_4D_Z2Z.restype = int
_faft256_4D.FAFT256_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double ]

cuda_faft128 = _faft256_4D.FAFT256_4D_Z2Z

# FAFT 64-points
_faft64_4D = ctypes.cdll.LoadLibrary( DIR_BASE_FAFT+'FAFT64_4D_Z2Z.so' )
_faft64_4D.FAFT64_4D_Z2Z.restype = int
_faft64_4D.FAFT64_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double ]
_faft64_4D.IFAFT64_4D_Z2Z.restype = int
_faft64_4D.IFAFT64_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double ]

faft64  = _faft64_4D.FAFT64_4D_Z2Z
ifaft64 = _faft64_4D.IFAFT64_4D_Z2Z

#



#..................................................................


class GPU_WignerDirac4D:
	"""
	
	"""
	def __init__( self ):


		size = self.gridDIM_y*self.gridDIM_p_y*self.gridDIM_x*self.gridDIM_p_x

		self.FAFT_axes0 = 0
		self.FAFT_axes1 = 1
		self.FAFT_axes2 = 2
		self.FAFT_axes3 = 3

		#m = size

		self.FAFT_segment_axes0 = 0
		self.FAFT_segment_axes1 = 0
		self.FAFT_segment_axes2 = 0
		self.FAFT_segment_axes3 = 0

		self.NF = 1

		# Phase space step size 
		self.dp_y  = 2*self.p_y_amplitude/float(self.gridDIM_p_y)           #axis 0
		self.dy    = 2*self.y_amplitude  /float(self.gridDIM_y)             #axis 1
		self.dp_x  = 2*self.p_x_amplitude/float(self.gridDIM_p_x)           #axis 2
		self.dx    = 2*self.x_amplitude  /float(self.gridDIM_x)             #axis 3  

		# Ambiguity space step size
		self.dtheta_y  = 2*self.theta_y_amplitude /float(self.gridDIM_y)     #axis 0
		self.dlambda_y = 2*self.lambda_y_amplitude/float(self.gridDIM_p_y)   #axis 1
		self.dtheta_x  = 2*self.theta_x_amplitude /float(self.gridDIM_x)     #axis 2
		self.dlambda_x = 2*self.lambda_x_amplitude/float(self.gridDIM_p_x)   #axis 3

		# delta parameters
		self.delta_p_y =   self.dp_y*self.dtheta_y /(2*np.pi)     #axis 0
		self.delta_y  =    self.dy  *self.dlambda_y/(2*np.pi)     #axis 1
		self.delta_p_x =   self.dp_x*self.dtheta_x /(2*np.pi)     #axis 2
		self.delta_x  =    self.dx  *self.dlambda_x/(2*np.pi)     #axis 3

		# Phase space
		self.p_y_range = np.linspace( -self.p_y_amplitude, self.p_y_amplitude-self.dp_y,  self.gridDIM_p_y)    #axis 0
		self.y_range   = np.linspace( -self.y_amplitude,   self.y_amplitude  -self.dy,    self.gridDIM_y  )    #axis 1
		self.p_x_range = np.linspace( -self.p_x_amplitude, self.p_x_amplitude-self.dp_x,  self.gridDIM_p_x)    #axis 2
		self.x_range   = np.linspace( -self.x_amplitude,   self.x_amplitude  -self.dx,    self.gridDIM_x  )    #axis 3

		# Ambiguity space range
		self.theta_y_range = np.linspace(-self.theta_y_amplitude,self.theta_y_amplitude-self.dtheta_y,self.gridDIM_y  )    #0
		self.lambda_y_range= np.linspace(-self.lambda_y_amplitude,self.lambda_y_amplitude-self.dlambda_y, self.gridDIM_p_y)#1
		self.theta_x_range = np.linspace(-self.theta_x_amplitude,self.theta_x_amplitude -self.dtheta_x,  self.gridDIM_x  ) #2
		self.lambda_x_range= np.linspace(-self.lambda_x_amplitude,self.lambda_x_amplitude-self.dlambda_x,self.gridDIM_p_x) #3

		# Grid 
		self.y   =   self.y_range[ np.newaxis, :, np.newaxis, np.newaxis ]   #axis 1
		self.p_y = self.p_y_range[ :, np.newaxis, np.newaxis, np.newaxis ]   #axis 0
		self.p_x = self.p_x_range[ np.newaxis, np.newaxis, :, np.newaxis ]   #axis 2
		self.x   =   self.x_range[ np.newaxis, np.newaxis, np.newaxis, : ]   #axis 3
		
		self.CUDA_constants =  '\n'
		self.CUDA_constants += '__device__ double c      = %f;   '%self.c
		self.CUDA_constants += '__device__ double HBar   = %f;   '%self.HBar
		self.CUDA_constants += '__device__ double dt   = %f;   '%self.dt
		self.CUDA_constants += '__device__ double mass = %f; \n'%self.mass

		self.CUDA_constants += '__device__ double dp_y   = %f; '%self.dp_y
		self.CUDA_constants += '__device__ double dy     = %f; '%self.dy
		self.CUDA_constants += '__device__ double dp_x   = %f; '%self.dp_x
		self.CUDA_constants += '__device__ double dx     = %f; \n'%self.dx

		self.CUDA_constants += '__device__ double dtheta_y   = %f; '%self.dtheta_y
		self.CUDA_constants += '__device__ double dlambda_y  = %f; '%self.dlambda_y
		self.CUDA_constants += '__device__ double dtheta_x   = %f; '%self.dtheta_x
		self.CUDA_constants += '__device__ double dlambda_x  = %f; \n'%self.dlambda_x

		self.CUDA_constants += '__device__ int gridDIM_x = %d; '%self.gridDIM_x
		self.CUDA_constants += '__device__ int gridDIM_y = %d; '%self.gridDIM_y

		try :
			self.CUDA_constants += '__device__ double D_lambda_x = %f; '%self.D_lambda_x
			self.CUDA_constants += '__device__ double D_lambda_y = %f; '%self.D_lambda_y
			self.CUDA_constants += '__device__ double D_theta_x  = %f; '%self.D_theta_x
			self.CUDA_constants += '__device__ double D_theta_y  = %f; \n'%self.D_theta_y
		except AttributeError:
			pass

		self.CUDA_constants +=  '\n'		

		print self.CUDA_constants

		#...........................................................................

		print '         GPU memory Total               ', pycuda.driver.mem_get_info()[1]/float(2**30) , 'GB'
		print '         GPU memory Free  (Before)      ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'

		self.W11_init_gpu = gpuarray.zeros(
				( self.gridDIM_p_y, self.gridDIM_y, self.gridDIM_p_x, self.gridDIM_x ), dtype=np.complex128 )

		
		self.W12_init_gpu = gpuarray.zeros_like(self.W11_init_gpu)
		self.W13_init_gpu = gpuarray.zeros_like(self.W11_init_gpu)
		self.W14_init_gpu = gpuarray.zeros_like(self.W11_init_gpu)

		self.W22_init_gpu = gpuarray.zeros_like(self.W11_init_gpu)
		self.W23_init_gpu = gpuarray.zeros_like(self.W11_init_gpu)
		self.W24_init_gpu = gpuarray.zeros_like(self.W11_init_gpu)

		self.W33_init_gpu = gpuarray.zeros_like(self.W11_init_gpu)
		self.W34_init_gpu = gpuarray.zeros_like(self.W11_init_gpu)

		self.W44_init_gpu = gpuarray.zeros_like(self.W11_init_gpu)
		
		print '         GPU memory Free  (After)       ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'	

		#............................................................................

		indexUnpack_x_p_string = """
			int i_x   = i%gridDIM_x;
			int i_p_x = (i/gridDIM_x) % gridDIM_x;
			int i_y   = (i/(gridDIM_x*gridDIM_x)) % gridDIM_y;
			int i_p_y = i/(gridDIM_x*gridDIM_x*gridDIM_y);

			double x   = dx  *( i_x   - gridDIM_x/2 );
			double p_x = dp_x*( i_p_x - gridDIM_x/2 );
			double y   = dy  *( i_y   - gridDIM_y/2 );
			double p_y = dp_y*( i_p_y - gridDIM_y/2 );			
			"""

		indexUnpack_lambda_theta_string = """
			int i_x   = i%gridDIM_x;
			int i_p_x = (i/gridDIM_x) % gridDIM_x;
			int i_y   = (i/(gridDIM_x*gridDIM_x)) % gridDIM_y;
			int i_p_y = i/(gridDIM_x*gridDIM_x*gridDIM_y);

			double lambda_x  = dlambda_x * ( i_x   - gridDIM_x/2 );
			double theta_x   = dtheta_x  * ( i_p_x - gridDIM_x/2 );
			double lambda_y  = dlambda_y * ( i_y   - gridDIM_y/2 );
			double theta_y   = dtheta_y  * ( i_p_y - gridDIM_y/2 );			
			"""

		indexUnpack_lambda_p_string = """
			int i_x    = i%gridDIM_x;
			int i_p_x  = (i/gridDIM_x) % gridDIM_x;
			int i_y    = (i/(gridDIM_x*gridDIM_x)) % gridDIM_y;
			int i_p_y  = i/(gridDIM_x*gridDIM_x*gridDIM_y);

			double lambda_x   = dlambda_x*( i_x   - gridDIM_x/2 );
			double p_x        = dp_x     *( i_p_x - gridDIM_x/2 );
			double lambda_y   = dlambda_y*( i_y   - gridDIM_y/2 );
			double p_y        = dp_y     *( i_p_y - gridDIM_y/2 );			
			"""
		indexUnpack_x_theta_string = """
			int i_x   = i%gridDIM_x;
			int i_p_x = (i/gridDIM_x) % gridDIM_x;
			int i_y   = (i/(gridDIM_x*gridDIM_x)) % gridDIM_y;
			int i_p_y = i/(gridDIM_x*gridDIM_x*gridDIM_y);

			double x       = dx      *( i_x   - gridDIM_x/2 );
			double theta_x = dtheta_x*( i_p_x - gridDIM_x/2 );
			double y       = dy      *( i_y   - gridDIM_y/2 );
			double theta_y = dtheta_y*( i_p_y - gridDIM_y/2 );	
			"""

		#...............................................................................................		

		self.Gaussian_GPU = ElementwiseKernel(
		    """pycuda::complex<double> *W , 
				double    mu_p_y, double    mu_y, double    mu_p_x, double    mu_x , 
				double sigma_p_y, double sigma_y, double sigma_p_x, double sigma_x """
		    ,
		    indexUnpack_x_p_string + """
			double temp =   exp(-0.5*( x   - mu_x   )*( x   - mu_x   )/( sigma_x   * sigma_x   )  );
			       temp *=  exp(-0.5*( y   - mu_y   )*( y   - mu_y   )/( sigma_y   * sigma_y   )  );
			       temp *=	exp(-0.5*( p_x - mu_p_x )*( p_x - mu_p_x )/( sigma_p_x * sigma_p_x )  );
			       temp *=	exp(-0.5*( p_y - mu_p_y )*( p_y - mu_p_y )/( sigma_p_y * sigma_p_y )  );

			W[i] = pycuda::complex<double>(  temp , 0. ); 

					"""
		    ,"Gaussian",  preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants )	

		#
		self.HOscillatorGound_GPU = ElementwiseKernel(
		    """pycuda::complex<double> *W, 
			double   x_mu, double    y_mu,
                        double p_x_mu, double  p_y_mu,
			double omega_x, double omega_y, double mass"""
		   ,
		   indexUnpack_x_p_string + """
			double temp  = (mass*pow( omega_x*(x-x_mu) ,2) + pow(p_x-p_x_mu,2)/mass)/omega_x;   
			       temp += (mass*pow( omega_y*(y-y_mu) ,2) + pow(p_y-p_y_mu,2)/mass)/omega_y;

			W[i] = pycuda::complex<double>(  exp(-temp) , 0. ); 
					          """
		   ,"Gaussian",  preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants )	

		# ...................................................................................................
		# Kinetic propagator ................................................................................
		#....................................................................................................

		kineticStringC = """ 
		__device__ double Omega(double p_x, double p_y, double m){ return  c/HBar*sqrt(pow(m*c,2)+pow(p_x,2)+pow(p_y,2)); }"""
  		
		kineticStringC += """
__device__ pycuda::complex<double> K11(double p_x, double p_y, double m, double dt){ 
 return  pycuda::complex<double>(cos(dt*Omega(p_x,p_y,m)),-((pow(c,2)*m*sin(dt*Omega(p_x,p_y,m)))/(Omega(p_x,p_y,m)*HBar)));}

__device__ pycuda::complex<double> K22(double p_x, double p_y, double m, double dt){ 
 return  pycuda::complex<double>(cos(dt*Omega(p_x,p_y,m)),(pow(c,2)*m*sin(dt*Omega(p_x,p_y,m)))/(Omega(p_x,p_y,m)*HBar));}

__device__ pycuda::complex<double> K14(double p_x, double p_y, double m, double dt){ 
 return  pycuda::complex<double>(-((c*p_y*sin(dt*Omega(p_x,p_y,m)))/(Omega(p_x,p_y,m)*HBar)),-((c*p_x*sin(dt*Omega(p_x,p_y,m)))/(Omega(p_x,p_y,m)*HBar)));}

__device__ pycuda::complex<double> K23(double p_x, double p_y, double m, double dt){ 
 return  pycuda::complex<double>((c*p_y*sin(dt*Omega(p_x,p_y,m)))/(Omega(p_x,p_y,m)*HBar),-((c*p_x*sin(dt*Omega(p_x,p_y,m)))/(Omega(p_x,p_y,m)*HBar)));} 
"""

		self.exp_p_lambda_plus_GPU = ElementwiseKernel(
 """pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
                                  pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
							        pycuda::complex<double> *W33, pycuda::complex<double> *W34,
											      pycuda::complex<double> *W44"""
			,
			indexUnpack_lambda_p_string + """ 				
 			double p_x_plus  = p_x + 0.5*HBar*lambda_x;
			double p_y_plus  = p_y + 0.5*HBar*lambda_y;
			
			pycuda::complex<double> k11 = K11(p_x_plus,p_y_plus,mass,dt);
			pycuda::complex<double> k22 = K22(p_x_plus,p_y_plus,mass,dt);
			pycuda::complex<double> k14 = K14(p_x_plus,p_y_plus,mass,dt);  
			pycuda::complex<double> k23 = K23(p_x_plus,p_y_plus,mass,dt);

			pycuda::complex<double> W11_, W12_, W13_, W14_,  W22_, W23_, W24_, W33_, W34_, W44_;

			W11_=k14*pycuda::conj<double>(W14[i]) + k11*W11[i];
			W12_=k14*pycuda::conj<double>(W24[i]) + k11*W12[i];
			W13_=k14*pycuda::conj<double>(W34[i]) + k11*W13[i];
			W14_=k11*W14[i] + k14*W44[i];
			W22_=k23*pycuda::conj<double>(W23[i]) + k11*W22[i];
			W23_=k11*W23[i] + k23*W33[i];
			W24_=k11*W24[i] + k23*W34[i];
			W33_=k14*W23[i] + k22*W33[i];
			W34_=k14*W24[i] + k22*W34[i];
			W44_=k23*W14[i] + k22*W44[i];

			W11[i] = W11_;
			W12[i] = W12_;
			W13[i] = W13_;
			W14[i] = W14_;

			W22[i] = W22_;
			W23[i] = W23_;
			W24[i] = W24_;

			W33[i] = W33_;
			W34[i] = W34_;

			W44[i] = W44_;	
			"""
  		       ,"exp_p_lambda_GPU",
			preamble = "#define _USE_MATH_DEFINES\n" + self.CUDA_constants +kineticStringC ) 


		self.exp_p_lambda_minus_GPU = ElementwiseKernel(
 """pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
                                  pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
							        pycuda::complex<double> *W33, pycuda::complex<double> *W34,
											      pycuda::complex<double> *W44"""
			,
			indexUnpack_lambda_p_string + """ 				
 			double p_x_minus  = p_x - 0.5*HBar*lambda_x;
			double p_y_minus  = p_y - 0.5*HBar*lambda_y;
			
			pycuda::complex<double> k11 = K11(p_x_minus,p_y_minus,mass,-dt);
			pycuda::complex<double> k22 = K22(p_x_minus,p_y_minus,mass,-dt);
			pycuda::complex<double> k14 = K14(p_x_minus,p_y_minus,mass,-dt);  
			pycuda::complex<double> k23 = K23(p_x_minus,p_y_minus,mass,-dt);

			pycuda::complex<double> W11_, W12_, W13_, W14_,  W22_, W23_, W24_, W33_, W34_, W44_;
			
			W11_=k11*W11[i] + k23*W14[i];
			W12_=k11*W12[i] + k14*W13[i];
			W13_=k23*W12[i] + k22*W13[i];
			W14_=k14*W11[i] + k22*W14[i];
			W22_=k11*W22[i] + k14*W23[i];
			W23_=k23*W22[i] + k22*W23[i];
			W24_=k14*pycuda::conj<double>(W12[i]) + k22*W24[i];
			W33_=k23*pycuda::conj<double>(W23[i]) + k22*W33[i];
			W34_=k14*pycuda::conj<double>(W13[i]) + k22*W34[i];
			W44_=k14*pycuda::conj<double>(W14[i]) + k22*W44[i];

			W11[i] = W11_;
			W12[i] = W12_;
			W13[i] = W13_;
			W14[i] = W14_;

			W22[i] = W22_;
			W23[i] = W23_;
			W24[i] = W24_;

			W33[i] = W33_;
			W34[i] = W34_;

			W44[i] = W44_;	
			"""
  		       ,"exp_p_lambda_GPU", preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants +kineticStringC ) 

		# ....................................................................................................
		#  Potential propagator ..............................................................................
		# ....................................................................................................

		potentialStringC = '__device__ double V(double x, double y){ \n return '+self.potentialString+';\n}'
				

		self.exp_x_theta_GPU = ElementwiseKernel(
			""" pycuda::complex<double> *B """
			,
			indexUnpack_x_theta_string + """ 
			double phase  = dt*V(x-0.5*theta_x , y-0.5*theta_y) - dt*V( x+0.5*theta_x , y+0.5*theta_y );
			
			double  r  = exp( - dt*D_theta_y * theta_x*theta_x - dt*D_theta_y * theta_y*theta_y );

			B[i] *= pycuda::complex<double>( r*cos(phase), -r*sin(phase) );

			"""
  		       ,"exp_x_theta_GPU",
			preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants + potentialStringC ) 


		#......................................................................................................
		#
		#               Initial states 
		#
		#......................................................................................................

		gaussianPsi_ParticleUp = """\n
		__device__ pycuda::complex<double> psi1( double x, double y , double u_x, double u_y, double x_sigma, double y_sigma){
		double r = exp( -0.5*pow(x/x_sigma,2) -0.5*pow(y/y_sigma,2)  )*(c + sqrt( c*c + u_x*u_x + u_y*u_y ));
		double phase = mass*(x*u_x + y*u_y);
		return pycuda::complex<double>( r*cos(phase) , r*sin(phase) );}\n

		__device__ pycuda::complex<double> psi2( double x, double y , double u_x, double u_y, double x_sigma, double y_sigma){
		return pycuda::complex<double>( 0., 0. );}\n

		__device__ pycuda::complex<double> psi3( double x, double y , double u_x, double u_y, double x_sigma, double y_sigma){
		return pycuda::complex<double>( 0., 0. );}\n

		__device__ pycuda::complex<double> psi4( double x, double y , double u_x, double u_y, double x_sigma, double y_sigma){
		double phase = mass*(x*u_x + y*u_y);
		double r = exp( -0.5*pow(x/x_sigma,2) -0.5*pow(y/y_sigma,2)  );
		return pycuda::complex<double>( r*(cos(phase)*u_x - sin(phase)*u_y) , r*(sin(phase)*u_x + cos(phase)*u_y) );}
		\n
		 """

		self.WignerDiracGaussian_ParticleUp_GPU = ElementwiseKernel("""
pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
                              pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
							    pycuda::complex<double> *W33, pycuda::complex<double> *W34,
								 		          pycuda::complex<double> *W44,
		double u_x, double u_y, double x_mu, double y_mu, double x_sigma, double y_sigma"""
		,
		indexUnpack_x_theta_string + """
		double x_minus = x - 0.5*HBar*theta_x - x_mu;
		double y_minus = y - 0.5*HBar*theta_y - y_mu;
		double x_plus  = x + 0.5*HBar*theta_x - x_mu;
		double y_plus  = y + 0.5*HBar*theta_y - y_mu;
 
		W11[i] =                        psi1(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(  psi1(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma )    );

		W14[i] =                        psi1(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(  psi4(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma )    );	
	 
		W44[i] =                        psi4(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(  psi4(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma )    );

		 """,
		"WignerDiracGaussian_ParticleUp",
		preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants + gaussianPsi_ParticleUp )

		#.....................................................................................................
		#
		# 					Reduction Kernels 
		#
		# Ehrenfest theorems .................................................................................

		volume_Define = "\n#define dV  dx*dy*dp_x*dp_y \n "

		x_Define    = "\n#define x(i)    dx*( (i%gridDIM_x) - 0.5*gridDIM_x )\n"
		p_x_Define  = "\n#define p_x(i)  dp_x*( ((i/gridDIM_x) % gridDIM_x)-0.5*gridDIM_x)\n"

		y_Define    = "\n#define y(i)   dy  *( (i/(gridDIM_x*gridDIM_x)) % gridDIM_y  - 0.5*gridDIM_y)\n"
		p_y_Define  = "\n#define p_y(i) dp_y*(  i/(gridDIM_x*gridDIM_x*gridDIM_y) - 0.5*gridDIM_y )\n"

		p_x_p_y_Define = p_x_Define + p_y_Define + volume_Define
		phaseSpaceDefine =  p_x_Define + p_y_Define + x_Define + y_Define + volume_Define

		self.Average_x_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( x(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments= "pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES"+x_Define+self.CUDA_constants)

		self.Average_x_square_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( x(i)*x(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments= "pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES"+x_Define+self.CUDA_constants)

		self.Average_p_x_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( p_x(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments="pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES" +p_x_Define+self.CUDA_constants)

		self.Average_p_x_square_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( p_x(i)*p_x(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments="pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES" +p_x_Define+self.CUDA_constants)


		self.Average_y_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( y(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments= "pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES"+y_Define+self.CUDA_constants)

		self.Average_p_y_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( p_y(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments= "pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES"+p_y_Define+self.CUDA_constants)

		#
		self.Average_y_square_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( y(i)*y(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments= "pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES"+y_Define+self.CUDA_constants)

		self.Average_p_y_square_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( p_y(i)*p_y(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments="pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES" +p_y_Define+self.CUDA_constants)


		argumentWString = """pycuda::complex<double> *W11, pycuda::complex<double> *W12,
				     pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
				     pycuda::complex<double> *W22, pycuda::complex<double> *W23,
				     pycuda::complex<double> *W24, pycuda::complex<double> *W33,
				     pycuda::complex<double> *W34, pycuda::complex<double> *W44"""

		self.Average_Alpha_1_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr  = "2.*dV*pycuda::real<double>( W14[i] + W23[i] )",
        			arguments = "pycuda::complex<double> *W14, pycuda::complex<double> *W23",
				preamble  = "#define _USE_MATH_DEFINES" +volume_Define+self.CUDA_constants)

		self.Average_Alpha_2_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr  = "2.*dV*pycuda::imag<double>( W23[i] - W14[i] )",
        			arguments = "pycuda::complex<double> *W14, pycuda::complex<double> *W23",
				preamble  = "#define _USE_MATH_DEFINES" +volume_Define+self.CUDA_constants)

		# ........................................

		kineticString = self.kineticString.replace( 'p_x' , 'p_x(i)'  )
		kineticString =      kineticString.replace( 'p_y' , 'p_y(i)'  )
		potentialString = (self.potentialString.replace( 'x'   , 'x(i)'    )).replace( 'y'   , 'y(i)'    )
 		energyString = kineticString + "+" + potentialString


		print "\n"
		print energyString

		self.Energy_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>(("+energyString+")*dx*dy*dp_x*dp_y*W[i])",
        			arguments= "pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

	def Gaussian_CPU(x,mu,sigma):
		return np.exp( - (x-mu)**2/sigma**2/2.  )/(sigma*np.sqrt( 2*np.pi  ))

	def fftGaussian_CPU(p,mu,sigma):
		return np.exp(-1j*mu*p , dtype=np.float64 )*np.exp( - p**2*sigma**2/2. , dtype=np.float64 )	

		
	def Norm_GPU( self, W_gpu ):
		norm = gpuarray.sum(W_gpu).get()
		return norm *self.dx * self.dp_x * self.dy * self.dp_y 

	def Norm_4x4_GPU( self, W11, W22, W33, W44 ):
		norm  = gpuarray.sum(W11).get()
		norm += gpuarray.sum(W22).get()
		norm += gpuarray.sum(W33).get()
		norm += gpuarray.sum(W44).get()
		return norm *self.dx * self.dp_x * self.dy * self.dp_y 

	def Normalize_4x4_GPU(self, W11, W12, W13, W14, W22, W23, W24, W33, W34, W44):
		norm = self.Norm_4x4_GPU( W11, W22, W33, W44 )
		W11 /= norm 
		W12 /= norm
		W13 /= norm
		W14 /= norm
 
		W22 /= norm
		W23 /= norm
		W24 /= norm

		W33 /= norm
		W34 /= norm

		W44 /= norm

	def Average_x_4x4_GPU(self,W11,W22,W33,W44):
		avx  = self.Average_x_GPU(W11).get()
		avx += self.Average_x_GPU(W22).get() 
		avx += self.Average_x_GPU(W33).get()
		avx += self.Average_x_GPU(W44).get()
		return avx

	def Average_p_x_4x4_GPU(self,W11,W22,W33,W44):
		avx  = self.Average_p_x_GPU(W11).get()
		avx += self.Average_p_x_GPU(W22).get() 
		avx += self.Average_p_x_GPU(W33).get()
		avx += self.Average_p_x_GPU(W44).get()
		return avx

	def Average_y_4x4_GPU(self,W11,W22,W33,W44):
		avx  = self.Average_y_GPU(W11).get()
		avx += self.Average_y_GPU(W22).get() 
		avx += self.Average_y_GPU(W33).get()
		avx += self.Average_y_GPU(W44).get()
		return avx

	def Average_p_y_4x4_GPU(self,W11,W22,W33,W44):
		avx  = self.Average_p_y_GPU(W11).get()
		avx += self.Average_p_y_GPU(W22).get() 
		avx += self.Average_p_y_GPU(W33).get()
		avx += self.Average_p_y_GPU(W44).get()
		return avx

		

	#
	def FAFT_64_128(self, W_gpu):
		print ' FAFT '
		cuda_faft128( int(W_gpu.gpudata),  self.dx,   self.delta_x,    self.FAFT_segment_axes0, self.FAFT_axes0, self.NF  )
		cuda_faft128( int(W_gpu.gpudata),  self.dp_x, self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes1, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dy,   self.delta_y,    self.FAFT_segment_axes2, self.FAFT_axes2, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dp_y, self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes3, self.NF  )

	def iFAFT_64_128(self, W_gpu):
		cuda_faft128( int(W_gpu.gpudata), self.dx,   -self.delta_x,   self.FAFT_segment_axes0, self.FAFT_axes0, self.NF )
		cuda_faft128( int(W_gpu.gpudata), self.dp_x, -self.delta_p_x, self.FAFT_segment_axes1, self.FAFT_axes1, self.NF )
		cuda_faft64(  int(W_gpu.gpudata), self.dy,   -self.delta_y,   self.FAFT_segment_axes2, self.FAFT_axes2, self.NF )
		cuda_faft64(  int(W_gpu.gpudata), self.dp_y, -self.delta_p_y, self.FAFT_segment_axes3, self.FAFT_axes3, self.NF )

	#..................FAFT 64  128 ..........................................

	def Fourier_X_To_Lambda_64_128_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dx, self.delta_x,  self.FAFT_segment_axes1, self.FAFT_axes1, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dy, self.delta_y,  self.FAFT_segment_axes3, self.FAFT_axes2, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_Lambda_To_X_64_128_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dx, -self.delta_x,  self.FAFT_segment_axes1, self.FAFT_axes1, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dy, -self.delta_y,  self.FAFT_segment_axes3, self.FAFT_axes2, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_P_To_Theta_64_128_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dp_x, self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dp_y, self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes3, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_Theta_To_P_64_128_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dp_x, -self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dp_y, -self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes3, self.NF  )
		W_gpu /= W_gpu.size

	#................FAFT 64 64 .............................................

	def Fourier_X_To_Lambda_64_64_GPU(self, W_gpu ):
		faft64( int(W_gpu.gpudata), self.dx, self.delta_x, self.FAFT_segment_axes1, self.FAFT_axes1, self.NF )
		faft64( int(W_gpu.gpudata), self.dy, self.delta_y, self.FAFT_segment_axes1, self.FAFT_axes3, self.NF )
		W_gpu /= W_gpu.size

	def Fourier_Lambda_To_X_64_64_GPU(self, W_gpu ):
		faft64( int(W_gpu.gpudata), self.dx, -self.delta_x, self.FAFT_segment_axes1, self.FAFT_axes1, self.NF )
		faft64( int(W_gpu.gpudata), self.dy, -self.delta_y, self.FAFT_segment_axes1, self.FAFT_axes3, self.NF )
		W_gpu /= W_gpu.size

	def Fourier_P_To_Theta_64_64_GPU(self, W_gpu ):
		faft64(  int(W_gpu.gpudata),  self.dp_x, self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		faft64(  int(W_gpu.gpudata),  self.dp_y, self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes2, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_Theta_To_P_64_64_GPU(self, W_gpu ):
		faft64(  int(W_gpu.gpudata),  self.dp_x, -self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		faft64(  int(W_gpu.gpudata),  self.dp_y, -self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes2, self.NF  )
		W_gpu /= W_gpu.size
		
	def Fourier_X_To_Lambda_64_64_4x4_GPU(self, W11, W12, W13, W14, W22, W23, W24, W33, W34, W44 ):
		self.Fourier_X_To_Lambda_64_64_GPU( W11 )
		self.Fourier_X_To_Lambda_64_64_GPU( W12 )
		self.Fourier_X_To_Lambda_64_64_GPU( W13 )
		self.Fourier_X_To_Lambda_64_64_GPU( W14 )

		self.Fourier_X_To_Lambda_64_64_GPU( W22 )
		self.Fourier_X_To_Lambda_64_64_GPU( W23 )
		self.Fourier_X_To_Lambda_64_64_GPU( W24 )

		self.Fourier_X_To_Lambda_64_64_GPU( W33 )
		self.Fourier_X_To_Lambda_64_64_GPU( W34 )

		self.Fourier_X_To_Lambda_64_64_GPU( W44 )

	def Fourier_Lambda_To_X_64_64_4x4_GPU(self, W11, W12, W13, W14, W22, W23, W24, W33, W34, W44 ):
		self.Fourier_Lambda_To_X_64_64_GPU( W11 )
		self.Fourier_Lambda_To_X_64_64_GPU( W12 )
		self.Fourier_Lambda_To_X_64_64_GPU( W13 )
		self.Fourier_Lambda_To_X_64_64_GPU( W14 )

		self.Fourier_Lambda_To_X_64_64_GPU( W22 )
		self.Fourier_Lambda_To_X_64_64_GPU( W23 )
		self.Fourier_Lambda_To_X_64_64_GPU( W24 )

		self.Fourier_Lambda_To_X_64_64_GPU( W33 )
		self.Fourier_Lambda_To_X_64_64_GPU( W34 )

		self.Fourier_Lambda_To_X_64_64_GPU( W44 )

	def Fourier_Theta_To_P_64_64_4x4_GPU(self, W11, W12, W13, W14, W22, W23, W24, W33, W34, W44 ):
		self.Fourier_Theta_To_P_64_64_GPU( W11 )
		self.Fourier_Theta_To_P_64_64_GPU( W12 )
		self.Fourier_Theta_To_P_64_64_GPU( W13 )
		self.Fourier_Theta_To_P_64_64_GPU( W14 )

		self.Fourier_Theta_To_P_64_64_GPU( W22 )
		self.Fourier_Theta_To_P_64_64_GPU( W23 )
		self.Fourier_Theta_To_P_64_64_GPU( W24 )

		self.Fourier_Theta_To_P_64_64_GPU( W33 )
		self.Fourier_Theta_To_P_64_64_GPU( W34 )

		self.Fourier_Theta_To_P_64_64_GPU( W44 )	

	def Fourier_P_To_Theta_64_64_4x4_GPU(self, W11, W12, W13, W14, W22, W23, W24, W33, W34, W44 ):
		self.Fourier_P_To_Theta_64_64_GPU( W11 )
		self.Fourier_P_To_Theta_64_64_GPU( W12 )
		self.Fourier_P_To_Theta_64_64_GPU( W13 )
		self.Fourier_P_To_Theta_64_64_GPU( W14 )

		self.Fourier_P_To_Theta_64_64_GPU( W22 )
		self.Fourier_P_To_Theta_64_64_GPU( W23 )
		self.Fourier_P_To_Theta_64_64_GPU( W24 )

		self.Fourier_P_To_Theta_64_64_GPU( W33 )
		self.Fourier_P_To_Theta_64_64_GPU( W34 )

		self.Fourier_P_To_Theta_64_64_GPU( W44 )
		
	#.......................................................................
	
	def Run(self):

		if self.gridDIM_x==128 and self.gridDIM_y == 64:
			Fourier_X_To_Lambda = self.Fourier_X_To_Lambda_64_128_GPU
			Fourier_Lambda_To_X = self.Fourier_Lambda_To_X_64_128_GPU
			Fourier_P_To_Theta  = self.Fourier_P_To_Theta_64_128_GPU
			Fourier_Theta_To_P  = self.Fourier_Theta_To_P_64_128_GPU

		elif self.gridDIM_x==64 and self.gridDIM_y == 64:
			Fourier_X_To_Lambda = self.Fourier_X_To_Lambda_64_64_4x4_GPU
			Fourier_Lambda_To_X = self.Fourier_Lambda_To_X_64_64_4x4_GPU
			Fourier_P_To_Theta  = self.Fourier_P_To_Theta_64_64_4x4_GPU
			Fourier_Theta_To_P  = self.Fourier_Theta_To_P_64_64_4x4_GPU


		try :
			import os
			os.remove (self.fileName)
		except OSError:
			pass


		self.file = h5py.File(self.fileName)

		#self.WriteHDF5_variables()
		#self.file.create_dataset('Hamiltonian', data = self.Hamiltonian.real )


		print '         GPU memory Total       ', pycuda.driver.mem_get_info()[1]/float(2**30) , 'GB'
		print '         GPU memory Free        ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'

		timeRangeIndex = range(0, self.timeSteps+1)

		W11 = self.W11_init_gpu
		W12 = self.W12_init_gpu
		W13 = self.W13_init_gpu
		W14 = self.W14_init_gpu

		W22 = self.W22_init_gpu
		W23 = self.W23_init_gpu
		W24 = self.W24_init_gpu

		W33 = self.W33_init_gpu
		W34 = self.W34_init_gpu
	
		W44 = self.W44_init_gpu

		average_x   = []
		average_p_x = []

		average_x_square   = []
		average_p_x_square = []
		average_y_square   = []
		average_p_y_square = []

		average_y   = []
		average_p_y = []

		average_Alpha_1 = []
		average_Alpha_2 = []		

		energy      = []


		for tIndex in timeRangeIndex:

			print ' t index = ', tIndex

			self.Normalize_4x4_GPU( W11, W12, W13, W14, W22, W23, W24, W33, W34, W44 ) 
            			    
			average_x.append(   self.Average_x_4x4_GPU  (W11, W22, W33, W44)   )
			average_p_x.append( self.Average_p_x_4x4_GPU(W11, W22, W33, W44)   )
			
			average_y.append(   self.Average_y_4x4_GPU  (W11, W22, W33, W44)   )
			average_p_y.append( self.Average_p_y_4x4_GPU(W11, W22, W33, W44)   )

			average_Alpha_1.append( self.Average_Alpha_1_GPU(W14,W23).get()  )	
			average_Alpha_2.append( self.Average_Alpha_2_GPU(W14,W23).get()  )		

			# p x  ->  p lambda
			Fourier_X_To_Lambda(W11, W12, W13, W14, W22, W23, W24, W33, W34, W44 ) 

			
			self.exp_p_lambda_plus_GPU (  W11, W12, W13, W14, W22, W23, W24, W33, W34, W44 ) 
			self.exp_p_lambda_minus_GPU ( W11, W12, W13, W14, W22, W23, W24, W33, W34, W44 ) 

			# p lambda  ->  p x
			Fourier_Lambda_To_X( W11, W12, W13, W14, W22, W23, W24, W33, W34, W44 ) 

			

		self.average_x   = np.array(average_x  )
		self.average_p_x = np.array(average_p_x)
		self.average_y   = np.array(average_y  )
		self.average_p_y = np.array(average_p_y)

		self.average_x_square   = np.array(average_x_square  )
		self.average_p_x_square = np.array(average_p_x_square)
		self.average_y_square   = np.array(average_y_square  )
		self.average_p_y_square = np.array(average_p_y_square)

		self.average_Alpha_1 = np.array(average_Alpha_1).flatten()
		self.average_Alpha_2 = np.array(average_Alpha_2).flatten()

		self.energy      = np.array(energy)

		self.file['/Ehrenfest/energy']       = self.energy
		self.file['/Ehrenfest/average_x']    = self.average_x
		self.file['/Ehrenfest/average_p_x']  = self.average_p_x
		self.file['/Ehrenfest/average_y']    = self.average_y
		self.file['/Ehrenfest/average_p_y']  = self.average_p_y

		self.file['/Ehrenfest/average_x_square'  ]  = self.average_x_square
		self.file['/Ehrenfest/average_p_x_square']  = self.average_p_x_square
		self.file['/Ehrenfest/average_y_square'  ]  = self.average_y_square
		self.file['/Ehrenfest/average_p_y_square']  = self.average_p_y_square

			






