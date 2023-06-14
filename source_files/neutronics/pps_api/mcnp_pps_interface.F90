module mcnp_pps
use parametric_plasma_source
implicit none

    ! Variables that need to be initialised 
    integer, parameter :: kb = 8 
    real(dknd) :: a(1:2**kb)
    real(dknd) :: s(1:2**kb)
    integer, allocatable :: cells(:)

contains

  subroutine pps_sample

    use mcnp_debug, only: rdum, idum
    use pblcom, only: pbl
    use mcnp_interfaces_mod, only : namchg, findlv
    implicit none
  
    integer :: i
    logical :: found
    logical :: first_call=.True.

    ! If this is the first time it is called initialise the source
    if(first_call) then
      call set_a_and_s(rdum(4),rdum(8),kb,a,s)
 
      ! Find cell indexes of cells named on the idum card
      allocate(cells(1:idum(2)))
      do i = 1,idum(2)
        cells(i) = namchg(1,idum(i+2))
      end do   

      first_call = .False.
    end if
     
    do while (.not. found)
    
      ! Sample position and energy of particle
      call sample_plasma(int(rdum(1)),rdum(2),rdum(3),rdum(4),rdum(5),rdum(6),rdum(7),rdum(8),rdum(9),rdum(10),rdum(11),random_func, int(rdum(14)),rdum(15),rdum(16),rdum(17),int(rdum(18)), kb, a, s, pbl%i%ipt, pbl%r%wgt, pbl%r%tme, pbl%r%erg, pbl%r%x, pbl%r%y, pbl%r%z)

      ! Find cell and reject if not in list
      pbl%i%icl = 0      
      call findlv
      found = .false.
      do i = 1, idum(2)
        if(pbl%i%icl == cells(i)) then
          found = .true.
          exit
        end if
      end do  
    end do      

    return
    
  end subroutine pps_sample

  function random_func()
    use mcnp_random, only: rang
    implicit none
   
   real (dknd) :: random_func
    random_func = rang()
  end function
  
end module mcnp_pps