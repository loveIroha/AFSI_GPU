! Scalar reconstruction from IBAMR advect_predictors3d.f.m4.
! Copyright (c) 2014-2026 by the IBAMR developers. BSD-3-Clause.
! See third_party/IBAMR_COPYRIGHT. Oracle only, never used by simulation.
double precision function median(a,b,c)
implicit none
double precision, intent(in) :: a,b,c
double precision lo,hi
lo = merge(a,b,a<=b)
hi = merge(b,a,a<=b)
median = max(lo,min(hi,c))
return
end
subroutine monotonize(Q,Q_L,Q_R,Q_star_L,Q_star_R)
implicit none
double precision Q(-1:1),Q_L,Q_R,Q_L_tmp,Q_R_tmp,Q_star_L,Q_star_R
double precision median
Q_L_tmp = median(Q(0),Q_L,Q(-1))
Q_R_tmp = median(Q(0),Q_R,Q(+1))
Q_star_L = median(Q(0),Q_L_tmp,3.d0*Q(0)-2.d0*Q_R_tmp)
Q_star_R = median(Q(0),Q_R_tmp,3.d0*Q(0)-2.d0*Q_L_tmp)
return
end
double precision function WENO5_interp(Q)
implicit none
double precision Q(-2:2)
double precision f(0:2)
double precision IS(0:2)
double precision omega_bar(0:2)
double precision omega(0:2),omega_sum
double precision alpha(0:2),alpha_sum
integer i
f(0) = (11.d0*Q( 0)-7.d0*Q(-1)+2.d0*Q(-2))/6.d0
f(1) = ( 2.d0*Q(+1)+5.d0*Q( 0)-     Q(-1))/6.d0
f(2) = (-1.d0*Q(+2)+5.d0*Q(+1)+2.d0*Q( 0))/6.d0
IS(0) = (13.d0/12.d0)*((Q( 0)-2.d0*Q(-1)+Q(-2))**2.d0) + 0.25d0*((3.d0*Q( 0)-4.d0*Q(-1)+     Q(-2))**2.d0)
IS(1) = (13.d0/12.d0)*((Q(+1)-2.d0*Q( 0)+Q(-1))**2.d0) + 0.25d0*((     Q(+1)-                Q(-1))**2.d0)
IS(2) = (13.d0/12.d0)*((Q(+2)-2.d0*Q(+1)+Q( 0))**2.d0) + 0.25d0*((     Q(+2)-4.d0*Q(+1)+3.d0*Q( 0))**2.d0)
omega_bar(0) = 0.1d0
omega_bar(1) = 0.6d0
omega_bar(2) = 0.3d0
do i = 0,2
alpha(i) = omega_bar(i)/(IS(i)+1.d-40)
enddo
alpha_sum = 0.d0
do i = 0,2
alpha_sum = alpha_sum + alpha(i)
enddo
do i = 0,2
omega(i) = alpha(i)/alpha_sum
enddo
do i = 0,2
omega(i) = omega(i)*(omega_bar(i)+omega_bar(i)**2.d0 -3.d0*omega_bar(i)*omega(i)+omega(i)**2.d0)/ (omega_bar(i)**2.d0+omega(i)*(1.d0-2.d0*omega_bar(i)))
enddo
omega_sum = 0.d0
do i = 0,2
omega_sum = omega_sum + omega(i)
enddo
do i = 0,2
omega(i) = omega(i)/omega_sum
enddo
WENO5_interp = 0.d0
do i = 0,2
WENO5_interp = WENO5_interp + omega(i)*f(i)
enddo
return
end
subroutine xsppm7_oracle(n,Q,lo,hi) bind(C,name="xsppm7_oracle")
use iso_c_binding
implicit none
integer(c_int),value :: n
real(c_double) :: Q(-4:n+3),lo(0:n),hi(0:n)
real(c_double) :: dQ(-2:n+1),Q_L(-1:n),Q_R(-1:n)
real(c_double) :: QQ,QQ_L,QQ_R,QQ_star_L,QQ_star_R,QQ_WENO(-2:2)
real(c_double) :: QQ_WENO_L,QQ_WENO_R,QQ_4th_L,QQ_4th_R
real(c_double) :: dQQ_C,dQQ_L,dQQ_R,dQQ
real(c_double) :: median,WENO5_interp
integer :: ic0,i
do ic0 = -2,n+1
dQQ_C = 0.5d0*(Q(ic0+1)-Q(ic0-1))
dQQ_L =       (Q(ic0  )-Q(ic0-1))
dQQ_R =       (Q(ic0+1)-Q(ic0  ))
if (dQQ_R*dQQ_L .gt. 1.d-12) then
dQQ = min(abs(dQQ_C),2.d0*abs(dQQ_L),2.d0*abs(dQQ_R))* sign(1.d0,dQQ_C)
else
dQQ = 0.d0
endif
dQ(ic0) = dQQ
enddo
do ic0 = -1,n
QQ   = Q(ic0)
QQ_L = (1.d0/420.d0)*( -   3.d0*Q(ic0+3) +  25.d0*Q(ic0+2) - 101.d0*Q(ic0+1) + 319.d0*Q(ic0  ) + 214.d0*Q(ic0-1) -  38.d0*Q(ic0-2) +   4.d0*Q(ic0-3))
QQ_R = (1.d0/420.d0)*( -   3.d0*Q(ic0-3) +  25.d0*Q(ic0-2) - 101.d0*Q(ic0-1) + 319.d0*Q(ic0  ) + 214.d0*Q(ic0+1) -  38.d0*Q(ic0+2) +   4.d0*Q(ic0+3))
Q_L(ic0) = QQ_L
Q_R(ic0) = QQ_R
call monotonize( Q(ic0-1), QQ_L,QQ_R,QQ_star_L,QQ_star_R)
if ( ((QQ_star_L-QQ_L)**2.d0 .ge. 1.d-12) .or. ((QQ_star_R-QQ_R)**2.d0 .ge. 1.d-12) ) then
do i = -2,2
QQ_WENO(i) = Q(ic0-i)
enddo
QQ_WENO_L = WENO5_interp(QQ_WENO)
do i = -2,2
QQ_WENO(i) = Q(ic0+i)
enddo
QQ_WENO_R = WENO5_interp(QQ_WENO)
if ( ((QQ_star_L-QQ)**2.d0 .le. 1.d-12) .or. ((QQ_star_R-QQ)**2.d0 .le. 1.d-12) ) then
QQ_WENO_L = median(QQ,QQ_WENO_L,QQ_L)
QQ_WENO_R = median(QQ,QQ_WENO_R,QQ_R)
call monotonize( Q(ic0-1), QQ_WENO_L,QQ_WENO_R,QQ_star_L,QQ_star_R)
else
QQ_4th_L = 0.5d0*( Q(ic0-1)+Q(ic0)) - (1.d0/6.d0)*( dQ(ic0)-dQ(ic0-1))
QQ_4th_R = 0.5d0*( Q(ic0)+Q(ic0+1)) - (1.d0/6.d0)*( dQ(ic0+1)-dQ(ic0))
QQ_4th_L = median(QQ_4th_L,QQ_WENO_L,QQ_L)
QQ_4th_R = median(QQ_4th_R,QQ_WENO_R,QQ_R)
call monotonize( Q(ic0-1), QQ_4th_L,QQ_4th_R,QQ_star_L,QQ_star_R)
endif
Q_L(ic0) = median(QQ_WENO_L,QQ_star_L,QQ_L)
Q_R(ic0) = median(QQ_WENO_R,QQ_star_R,QQ_R)
endif
enddo

do i=0,n
lo(i)=Q_R(i-1)
hi(i)=Q_L(i)
enddo
end subroutine
